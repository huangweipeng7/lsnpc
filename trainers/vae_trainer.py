"""VAE-based trainer for noisy label correction."""

from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from .base_trainer import Trainer


class VAETrainer(Trainer):
    """VAE-based trainer for noisy label correction.

    Args:
        n_labels: Number of labels.
        model: VAE model to train.
        loss_fn: Loss function.
        optimizer: PyTorch optimizer.
        arg_dict: Arguments dictionary.
        lr_scheduler: Learning rate scheduler.
        train_on_val: Whether to train on validation.
        eval_test_at_final_loop_only: Whether to evaluate test only at final epoch.
        grad_norm: Gradient clipping norm. Default: 2.
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
        train_on_val: bool = True,
        eval_test_at_final_loop_only: bool = True,
        grad_norm: int = 2,
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
            train_on_val=train_on_val,
            eval_test_at_final_loop_only=eval_test_at_final_loop_only,
            metric_storing_path=metric_storing_path,
            accelerator=accelerator
        )

        self.grad_norm = grad_norm

    def train_one_epoch(
        self,
        train_loader,
        epoch: int = -1
    ) -> float:
        """Train VAE model for one epoch."""
        self.train()

        loss_all = 0.0
        n_runs = 0

        for batch in (pbar := tqdm.tqdm(train_loader)):
            data = batch['data']
            target_dist = self.get_target(batch)
            y_hat = torch.bernoulli(target_dist)

            self.optimizer.zero_grad() 
            res_doc = self.model(data, y_hat)
            loss = self.loss_fn(res_doc, y_hat)

            self.accelerator.backward(loss)

            loss_all += loss.item()
            n_runs += 1

            self.accelerator.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_norm
            )
            self.optimizer.step()

            pbar.set_description(f'training loss: {loss_all/n_runs:.4f}')

        return loss_all / n_runs

    def train_on_val_one_epoch(self, train_loader) -> float:
        """Train on validation set for one epoch."""
        self.train()

        loss_all = 0.0
        n_runs = 0

        for batch in (pbar := tqdm.tqdm(train_loader)):
            data = batch['data']
            y_hat = torch.bernoulli(self.get_target(batch))

            self.optimizer.zero_grad()
            y_pred = self.model(data, y_hat)
            loss = self.loss_fn(y_pred, y_hat)

            self.accelerator.backward(loss)

            loss_all += loss.item()
            n_runs += 1

            self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=10)
            self.optimizer.step()

            pbar.set_description(f'training loss on validation: {loss_all/n_runs:.4f}')

        return loss_all / n_runs

    @torch.inference_mode()
    def predict(
        self,
        batch: Dict,
        sample_type: str = 'map',
        n_samples: int = 8,
        **kwargs
    ) -> torch.Tensor:
        """Make predictions with Monte Carlo sampling."""
        data = batch['data']
        target_dist = self.get_target(batch)

        self.model.eval()
        y = 0.0

        if sample_type == 'sample':
            for _ in range(n_samples):
                target = torch.bernoulli(target_dist)
                y += F.sigmoid(self.model(data, target)['y_logits']) / n_samples
        elif sample_type == 'mean':
            for _ in range(n_samples):
                y += F.sigmoid(self.model(data, target_dist)['y_logits']) / n_samples
        elif sample_type == 'map':
            y += (F.sigmoid(self.model(data, target_dist)['y_logits']) > 0.5).int()
        else:
            raise ValueError("Only 'sample', 'mean', and 'map' are supported.")

        return torch.clamp(y, min=0, max=1).float()

    def train(self) -> None:
        """Set model to training mode, pretrained classifier to eval."""
        self.model.train()
        if hasattr(self.model, 'pretrained_clf'):
            self.model.pretrained_clf.eval()

    def eval(self) -> None:
        """Set model to evaluation mode."""
        self.model.eval()

    @torch.inference_mode()
    def get_target(self, batch: Dict) -> torch.Tensor:
        """Get target distribution from pretrained classifier."""
        out = self.model.pretrained_clf(batch['data']) 
        return F.sigmoid(out if not isinstance(out, tuple) else out[0])
