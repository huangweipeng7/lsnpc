"""HLC (Hard Label Correction) trainer for iterative label refinement."""

import gc
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from .base_trainer import DEFAULT_GC_INTERVAL, Trainer

logger = logging.getLogger(__name__)


class HLCTrainer(Trainer):
    """Hard Label Correction trainer for iterative label refinement.

    Args:
        n_labels: Number of labels.
        model: PyTorch model to train.
        loss_fn: Loss function.
        optimizer: PyTorch optimizer.
        arg_dict: Dictionary of arguments.
        lr_scheduler: Learning rate scheduler.
        delta: Threshold for label correction. Default: 0.4.
        epoch_update_start: Epoch to start label updates. Default: 5.
        beta: Weight for prediction confidence. Default: 0.5.
        eval_test_at_final_loop_only: Whether to evaluate test only at final epoch.
        metric_storing_path: Path to store results.
        accelerator: Accelerator for distributed training.
    """

    def __init__(
        self,
        n_labels: int,
        model: nn.Module,
        loss_fn,
        optimizer: torch.optim.Optimizer,
        arg_dict: Dict,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        delta: float = 0.4,
        epoch_update_start: int = 5,
        beta: float = 0.5,
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

        self.delta = delta
        self.beta = beta
        self.epoch_update_start = epoch_update_start
        self.labels: List[List[float]] = []

    def train_model(
        self,
        n_epochs: int,
        train_loader,
        val_loader: Optional = None,
        test_loader: Optional = None,
        verbose: bool = False,
        clean_set_loader: Optional = None,
        gc_interval: int = DEFAULT_GC_INTERVAL
    ) -> None:
        """Training loop with dynamic label correction."""
        logger.info(f'Training with uid: {self.uid}')

        self.labels = []
        for batch in tqdm.tqdm(train_loader):
            self.labels.append(batch['labels'].tolist())
        logger.info('Labels loaded')

        for ep in range(n_epochs):
            logger.info(f'Epoch {ep}')
            if ep < self.epoch_update_start:
                train_loss = self.train_one_epoch(ep, train_loader)
            else:
                delta = self.delta * max(0, 0.2 * (10 - ep))
                train_loss, self.labels, corrected_num = self.train_one_epoch_hlc(
                    ep, delta, train_loader
                )

            self.eval_and_save(ep, n_epochs, val_loader, test_loader, verbose)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Periodic garbage collection to free memory
            if ep > 0 and ep % gc_interval == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                logger.debug(f'Garbage collection performed at epoch {ep}')

    def train_one_epoch(
        self,
        ep: int,
        train_loader
    ) -> float:
        """Train with original labels."""
        self.train()

        loss_all = 0.0
        n_runs = 0

        for i, batch in enumerate(pbar := tqdm.tqdm(train_loader)):
            data = batch['data'].to(self.device)
            target = torch.tensor(
                self.labels[i],
                dtype=torch.float32,
                device=self.device
            )

            self.optimizer.zero_grad()
            pred, _ = self.model(data)
            loss = self.loss_fn(pred, target)

            if self.accelerator is not None:
                self.accelerator.backward(loss)
            else:
                loss.backward()

            loss_all += loss.item()
            n_runs += 1

            if self.accelerator is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=2)
            else:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2)
            self.optimizer.step()

            pbar.set_description(f'training loss: {loss_all/n_runs:.4f}')

        return loss_all / n_runs

    def train_one_epoch_hlc(
        self,
        ep: int,
        delta: float,
        train_loader,
    ) -> Tuple[float, List[List[float]], int]:
        """Train with hard label correction."""
        self.train()

        loss_all = 0.0
        n_runs = 0
        corrected_targets = []
        corrected_num_total = 0

        for i, batch in enumerate(pbar := tqdm.tqdm(train_loader)):
            data = batch['data'].to(self.device)
            target = torch.tensor(
                self.labels[i],
                dtype=torch.float32,
                device=self.device
            )

            self.optimizer.zero_grad()
            pred, label_dependency = self.model(data)
            corrected_labels_batch = torch.zeros_like(target)

            batch_corrected = 0
            for j in range(pred.size(0)):
                t_pred = pred[j]
                t_num_labels = torch.nonzero(target[j]).size(0)
                t_noisy_labels = torch.nonzero(target[j]).squeeze()
                t_pred_labels = torch.topk(t_pred, int(t_num_labels)).indices

                original_sc = (
                    self.beta * torch.sum(torch.sigmoid(t_pred[t_noisy_labels]))
                    + (1 - self.beta) * self._compute_label_dependency(label_dependency[j], t_noisy_labels)
                )
                predicted_sc = (
                    self.beta * torch.sum(torch.sigmoid(t_pred[t_pred_labels]))
                    + (1 - self.beta) * self._compute_label_dependency(label_dependency[j], t_pred_labels)
                )

                SR = original_sc / predicted_sc if predicted_sc > 0 else torch.tensor(1.0)

                if SR <= delta:
                    corrected_labels_batch[j, t_pred_labels] = 1.0
                    batch_corrected += 1
                else:
                    corrected_labels_batch[j, t_noisy_labels] = 1.0

            loss = self.loss_fn(pred, corrected_labels_batch)

            if self.accelerator is not None:
                self.accelerator.backward(loss)
            else:
                loss.backward()

            if self.accelerator is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            else:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

            loss_all += loss.item()
            n_runs += 1
            corrected_targets.append(corrected_labels_batch.cpu().tolist())
            corrected_num_total += batch_corrected

            pbar.set_description(
                f'training loss: {loss_all/n_runs:.4f} | '
                f'HLC corrected {batch_corrected}/{pred.size(0)} (delta={delta})'
            )

        return loss_all / n_runs, corrected_targets, corrected_num_total

    def _compute_label_dependency(
        self,
        label_dependency: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """Compute label dependency score."""
        score = 0.0
        for j in range(labels.size(0)):
            for k in range(labels.size(0)):
                if labels[k] != labels[j]:
                    score += label_dependency[int(labels[j]), int(labels[k])]
        return score

    @torch.inference_mode()
    def predict(self, batch: Dict) -> torch.Tensor:
        """Make predictions without gradients."""
        data = batch['data'].to(self.device)
        pred, _ = self.model(data)
        return F.sigmoid(pred)
