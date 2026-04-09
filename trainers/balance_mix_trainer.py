"""Balanced Mixup trainer for handling class imbalance and noisy labels."""

import gc
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import tqdm
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader

from .base_trainer import DEFAULT_GC_INTERVAL, Trainer

logger = logging.getLogger(__name__)


class BalanceMixTrainer(Trainer):
    """Balanced Mixup trainer implementing two-sampler mixup with GMM-based label refinement."""

    def __init__(
        self,
        n_labels: int,
        model: nn.Module,
        loss_fn,
        optimizer: torch.optim.Optimizer,
        arg_dict: Dict,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        warmup_epochs: int = 5,
        alpha: float = 4.0,
        relabel_weight: float = 1.0,
        ambiguous_weight: float = 1.0,
        eval_test_at_final_loop_only: bool = False,
        metric_storing_path: Union[str, Path] = './runs/results.csv',
        accelerator=None
    ):
        super().__init__(
            n_labels=n_labels,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            arg_dict=arg_dict,
            lr_scheduler=lr_scheduler,
            eval_test_at_final_loop_only=eval_test_at_final_loop_only,
            metric_storing_path=metric_storing_path,
            accelerator=accelerator
        )

        self.warmup_epochs = warmup_epochs
        self.alpha = alpha
        self.relabel_weight = relabel_weight
        self.ambiguous_weight = ambiguous_weight

        self.batch_size: int = 0
        self.p_clean: Optional[torch.Tensor] = None
        self.clean_mask: Optional[torch.Tensor] = None
        self.labels: Optional[torch.Tensor] = None
        self.dataset = None
        self.prob_sampling: Optional[torch.Tensor] = None
        self.gmms_pos: List[Optional[GaussianMixture]] = []
        self.gmms_neg: List[Optional[GaussianMixture]] = []

    def train_model(
        self,
        n_epochs: int,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        verbose: bool = False,
        clean_set_loader: Optional[DataLoader] = None,
        gc_interval: int = DEFAULT_GC_INTERVAL
    ) -> None:
        """Main training loop with balanced mixup."""
        self.batch_size = train_loader.batch_size
        dataset = train_loader.dataset

        self.p_clean = torch.ones(len(dataset), self.n_labels, device=self.device)
        self.clean_mask = torch.ones(len(dataset), self.n_labels, dtype=torch.bool, device=self.device)
        self.labels = torch.zeros(len(dataset), self.n_labels, device=self.device)

        train_loader = DataLoader(
            WithIndices(dataset),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

        for batch in train_loader:
            batch_labels = batch['labels'].float().to(self.device)
            batch_indices = batch['index']
            self.labels[batch_indices] = batch_labels

        self.dataset = train_loader.dataset

        for epoch in range(n_epochs):
            logger.info(f'Epoch {epoch}')

            self.update_minority_sampling_probs(train_loader)

            if epoch < self.warmup_epochs:
                if epoch == 0:
                    logger.info('#' * 50)
                    logger.info("Warm-up stage (no label refinement)...")
                self.train_one_epoch(train_loader, epoch=epoch, alpha=self.alpha)
            else:
                if epoch == self.warmup_epochs:
                    logger.info('#' * 50)
                    logger.info("Full BalanceMix stage...")
                self.train_one_epoch(
                    train_loader,
                    epoch=epoch,
                    alpha=self.alpha,
                    label_refinement=True
                )

            self.fit_gmms_on_training_data(train_loader)

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

            if self.train_on_val:
                assert clean_set_loader is not None
                self.train_on_val_one_epoch(clean_set_loader)

            self.eval_and_save(epoch, n_epochs, val_loader, test_loader, verbose)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Periodic garbage collection to free memory
            if epoch > 0 and epoch % gc_interval == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                logger.debug(f'Garbage collection performed at epoch {epoch}')

    def train_one_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        alpha: float = 4.0,
        label_refinement: bool = False
    ) -> float:
        """Train with balanced mixup."""
        self.train()

        loss_all = 0.0
        n_runs = 0

        for batch in (pbar := tqdm.tqdm(train_loader)):
            self.optimizer.zero_grad()

            target = batch['labels'].float().to(self.device)

            if label_refinement:
                new_labels, ambiguous_mask, ratio_clean, ratio_relabel, ratio_ambiguous = \
                    self.gmm_label_refinement_three_state(epoch, batch)
                refined_target = new_labels.to(self.device)
            else:
                refined_target = target
                ambiguous_mask = None
                ratio_clean = ratio_relabel = ratio_ambiguous = 0.0

            imgs, lbs, weights = self.balancemix_two_sampler_batch(
                epoch, batch, refined_target, alpha=alpha,
                label_refinement=label_refinement,
                ambiguous_mask=ambiguous_mask
            )

            pred = self.model(imgs)
            loss = self.weighted_bce_loss(pred, lbs, weights=weights if label_refinement else None)

            self.accelerator.backward(loss)

            loss_all += loss.item()
            n_runs += 1

            self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=2)
            self.optimizer.step()

            if label_refinement:
                pbar.set_description(
                    f'training loss: {loss_all/n_runs:.4f} | '
                    f'clean: {ratio_clean:.4f}, relabel: {ratio_relabel:.4f}, '
                    f'ambiguous: {ratio_ambiguous:.4f}'
                )
            else:
                pbar.set_description(f'training loss: {loss_all/n_runs:.4f}')

        return loss_all / n_runs

    @torch.inference_mode()
    def update_minority_sampling_probs(
        self,
        train_loader: DataLoader,
        epsilon: float = 1e-6
    ) -> None:
        """Update sampling probabilities based on prediction confidence."""
        all_preds = []
        all_labels = []
        all_indices = []

        for batch in train_loader:
            probs = self.predict(batch)
            all_preds.append(probs)
            all_labels.append(batch['labels'])
            all_indices.append(batch['index'])

        all_preds = torch.cat(all_preds)
        all_indices = torch.cat(all_indices)
        all_labels = torch.cat(all_labels).to(self.device)

        per_label_conf = all_labels * all_preds + (1 - all_labels) * (1 - all_preds)
        score = per_label_conf.mean(dim=1)
        hardness = 1.0 / (score + epsilon)
        prob_sampling = hardness / hardness.sum()

        ordered_prob_sampling = torch.empty_like(prob_sampling)
        ordered_prob_sampling[all_indices] = prob_sampling
        self.prob_sampling = ordered_prob_sampling

        logger.debug('Updated minority sampling probabilities.')

    @torch.inference_mode()
    def gmm_label_refinement_three_state(
        self,
        epoch: int,
        batch: Dict,
        eps: float = 0.975
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float, float]:
        """Refine labels using GMM-based three-state classification."""
        self.eval()

        data = batch['data'].to(self.device)
        target = batch['labels'].float().to(self.device)
        clean_mask = self.clean_mask[batch['index']]

        unlabeled = ~clean_mask

        x_aug1 = torch.stack([self.rand_aug(data[i]) for i in range(data.size(0))])
        x_aug2 = torch.stack([self.rand_aug(data[i]) for i in range(data.size(0))])

        p1 = torch.sigmoid(self.model(x_aug1))
        p2 = torch.sigmoid(self.model(x_aug2))
        conf = 0.5 * (p1 + p2)

        new_labels = target.clone()

        mask1 = (conf > eps) & unlabeled
        new_labels[mask1] = 1.0

        mask0 = (conf < 1 - eps) & unlabeled
        new_labels[mask0] = 0.0

        ambiguous = unlabeled & ~(mask1 | mask0)

        ratio_clean = clean_mask.float().mean().item()
        ratio_relabel = (mask1.float().mean() + mask0.float().mean()).item()
        ratio_ambiguous = ambiguous.float().mean().item()

        self.labels[batch['index']] = new_labels.to(self.device)

        self.train()
        return new_labels, ambiguous, ratio_clean, ratio_relabel, ratio_ambiguous

    @torch.inference_mode()
    def fit_gmms_on_training_data(self, train_loader: DataLoader) -> None:
        """Fit GMMs on per-sample losses."""
        self.eval()

        labels = []
        probs = []
        indices = []

        for batch in train_loader:
            labels.append(batch['labels'].float().to(self.device))
            indices.append(batch['index'])
            probs.append(self.predict(batch))

        probs = torch.cat(probs, dim=0)
        labels = torch.cat(labels, dim=0)
        indices = torch.cat(indices, dim=0)

        loss_mat, loss_pos, loss_neg, pos_idx, neg_idx = self.compute_bce_losses(probs, labels)

        self.gmms_pos = self.fit_gmm(loss_pos)
        self.gmms_neg = self.fit_gmm(loss_neg)

        new_p_clean = self.compute_clean_prob(loss_mat, pos_idx, neg_idx)
        self.p_clean[indices] = new_p_clean
        self.clean_mask[indices] = self.classify_labels(self.p_clean[indices])

        logger.info(f'Fitted GMMs - clean ratio: {self.clean_mask.float().mean().item():.4f}')
        self.train()

    def balancemix_two_sampler_batch(
        self,
        epoch: int,
        batch: Dict,
        target: torch.Tensor,
        alpha: float = 4.0,
        label_refinement: bool = False,
        ambiguous_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Generate mixed batch using two-sampler strategy."""
        data = batch['data'].to(self.device)
        all_indices = list(range(len(target)))

        batch_d = random.sample(all_indices, k=min(self.batch_size, len(all_indices)))

        batch_m = random.choices(
            range(len(self.dataset)),
            weights=self.prob_sampling,
            k=min(self.batch_size, len(all_indices))
        )

        batch_d_tensor = torch.tensor(batch_d, device=self.device)

        x_m = torch.stack([self.dataset[i]['data'] for i in batch_m]).to(self.device)
        y_m = torch.stack([self.labels[i] for i in batch_m]).to(self.device)
        x_d = data[batch_d_tensor]
        y_d = target[batch_d_tensor]

        images, labels, weights = self.mix_batch(
            x_d, x_m, y_d, y_m, alpha, label_refinement, ambiguous_mask
        )

        return images, labels, weights

    @torch.inference_mode()
    def mix_batch(
        self,
        x1_batch: torch.Tensor,
        x2_batch: torch.Tensor,
        y1_batch: torch.Tensor,
        y2_batch: torch.Tensor,
        alpha: float = 4.0,
        label_refinement: bool = False,
        ambiguous_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Apply mixup to two batches."""
        batch_size = x1_batch.size(0)
        weights = None

        if alpha > 0:
            lam_values = np.random.beta(alpha, alpha, size=batch_size)
            lam_values = np.maximum(lam_values, 1 - lam_values)
            lam_tensor = torch.tensor(lam_values, dtype=torch.float32, device=x1_batch.device)
        else:
            lam_tensor = torch.ones(batch_size, device=x1_batch.device)

        lam_img = lam_tensor.view(batch_size, 1, 1, 1)
        lam_label = lam_tensor.view(batch_size, 1)

        mixed_images = lam_img * x1_batch + (1 - lam_img) * x2_batch
        mixed_labels = lam_label * y1_batch + (1 - lam_label) * y2_batch

        if label_refinement:
            self.eval()

            p1 = torch.sigmoid(self.model(x1_batch))
            loss_mat1, _, _, pos_idx1, neg_idx1 = self.compute_bce_losses(p1, y1_batch)
            p_clean_batch1 = self.compute_clean_prob(loss_mat1, pos_idx1, neg_idx1)
            w1 = self.ambiguous_weights(p_clean_batch1, ambiguous_mask)

            p2 = torch.sigmoid(self.model(x2_batch))
            loss_mat2, _, _, pos_idx2, neg_idx2 = self.compute_bce_losses(p2, y2_batch)
            p_clean_batch2 = self.compute_clean_prob(loss_mat2, pos_idx2, neg_idx2)
            w2 = self.ambiguous_weights(p_clean_batch2, ambiguous_mask)

            weights = lam_label * w1 + (1 - lam_label) * w2
            self.train()

        return mixed_images, mixed_labels, weights

    def compute_bce_losses(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, List[np.ndarray], List[np.ndarray], torch.Tensor, torch.Tensor]:
        """Compute per-sample BCE losses."""
        bce = nn.BCELoss(reduction='none')
        loss_mat = bce(probs, labels)

        pos_idx = (labels == 1)
        neg_idx = (labels == 0)

        loss_pos = [
            loss_mat[pos_idx[:, k], k].detach().cpu().numpy()
            for k in range(labels.shape[1])
        ]
        loss_neg = [
            loss_mat[neg_idx[:, k], k].detach().cpu().numpy()
            for k in range(labels.shape[1])
        ]

        return loss_mat, loss_pos, loss_neg, pos_idx, neg_idx

    def fit_gmm(self, loss_lists: List[np.ndarray]) -> List[Optional[GaussianMixture]]:
        """Fit GMMs to loss distributions."""
        gmms = []
        for losses in loss_lists:
            if len(losses) < 2:
                gmms.append(None)
                continue

            losses = np.array(losses).reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, max_iter=200, tol=1e-4, random_state=42)
            gmm.fit(losses)
            gmms.append(gmm)

        return gmms

    def compute_clean_prob(
        self,
        loss_mat: torch.Tensor,
        pos_idx: torch.Tensor,
        neg_idx: torch.Tensor
    ) -> torch.Tensor:
        """Compute probability of clean labels."""
        B, K = loss_mat.shape
        p_clean = torch.zeros_like(loss_mat)

        for k in range(K):
            idx = pos_idx[:, k]
            if self.gmms_pos[k] is not None and idx.any():
                losses = loss_mat[idx, k].detach().cpu().numpy().reshape(-1, 1)
                prob = self.gmms_pos[k].predict_proba(losses)
                small_comp = np.argmin(self.gmms_pos[k].means_)
                p_clean[idx, k] = torch.tensor(
                    prob[:, small_comp],
                    dtype=torch.float32,
                    device=loss_mat.device
                )

            idx = neg_idx[:, k]
            if self.gmms_neg[k] is not None and idx.any():
                losses = loss_mat[idx, k].detach().cpu().numpy().reshape(-1, 1)
                prob = self.gmms_neg[k].predict_proba(losses)
                small_comp = np.argmin(self.gmms_neg[k].means_)
                p_clean[idx, k] = torch.tensor(
                    prob[:, small_comp],
                    dtype=torch.float32,
                    device=loss_mat.device
                )

        return p_clean

    def ambiguous_weights(
        self,
        p_clean: torch.Tensor,
        ambiguous_mask: torch.Tensor,
        scaling_factor: float = 1.0
    ) -> torch.Tensor:
        """Compute weights for ambiguous samples."""
        w = torch.ones_like(p_clean)
        w[ambiguous_mask] = p_clean[ambiguous_mask] * scaling_factor
        return w

    def rand_aug(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RandAugment transformation."""
        to_pil = ToPILImage()
        transform = transforms.Compose([
            transforms.RandAugment(num_ops=1, magnitude=2),
            transforms.ToTensor()
        ])
        pil_img = to_pil(x.cpu())
        return transform(pil_img).to(x.device)

    def classify_labels(self, p_clean: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
        """Classify labels as clean or noisy."""
        return p_clean > thresh

    def weighted_bce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Weighted BCE loss supporting three states."""
        if weights is not None:
            return nn.functional.binary_cross_entropy_with_logits(
                logits, labels, weight=weights, reduction='mean'
            )
        else:
            return nn.functional.binary_cross_entropy_with_logits(
                logits, labels, reduction='mean'
            )


# Import at module level to avoid circular imports
from utils import WithIndices
