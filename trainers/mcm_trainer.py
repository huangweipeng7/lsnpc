"""MCM (Multi-Component Model) trainer with custom loss handling."""

import gc
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from .base_trainer import DEFAULT_GC_INTERVAL, Trainer

logger = logging.getLogger(__name__)


class MCMTrainer(Trainer):
    """Multi-Component Model trainer with custom loss handling."""

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
        """Training loop with manual CUDA cache clearing."""
        logger.info(f'Training with uid: {self.uid}')

        for epoch in range(n_epochs):
            logger.info(f'Epoch {epoch}')
            train_loss = self.train_one_epoch(train_loader, epoch=epoch)

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
        train_loader,
        epoch: int = -1
    ) -> float:
        """Train MCM model for one epoch."""
        self.train()

        loss_all = 0.0
        n_runs = 0

        for batch in (pbar := tqdm.tqdm(train_loader)):
            data = batch['data'].to(self.device)
            target = batch['labels'].float().to(self.device)

            self.optimizer.zero_grad()
            y_preds, noisy_probs = self.model(data)
            loss, _ = self.loss_fn(noisy_probs, target, self.model.pred_sigmoid(y_preds))

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

    @torch.inference_mode()
    def predict(self, batch: Dict) -> torch.Tensor:
        """Make predictions without gradients."""
        data = batch['data'].to(self.device)
        y, _ = self.model(data)
        return F.sigmoid(y)
