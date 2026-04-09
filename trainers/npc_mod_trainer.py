"""NPC Modified trainer extending VAETrainer."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import tqdm

from .vae_trainer import VAETrainer


class NPCModTrainer(VAETrainer):
    """NPC Modified trainer extending VAETrainer."""

    def train_one_epoch(
        self,
        train_loader,
        epoch: int = -1
    ) -> float:
        """Train NPC-MOD model for one epoch."""
        self.train()

        loss_all = 0.0
        n_runs = 0

        for batch in (pbar := tqdm.tqdm(train_loader)):
            data = batch['data'].to(self.device)
            target_dist = self.get_target(batch)
            target = torch.bernoulli(target_dist)

            self.optimizer.zero_grad()
            res_doc = self.model(data, target)
            loss = self.loss_fn(res_doc, target, target_dist)

            self.accelerator.backward(loss)

            loss_all += loss.item()
            n_runs += 1

            self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm)
            self.optimizer.step()

            pbar.set_description(f'training loss: {loss_all/n_runs:.4f}')

        return loss_all / n_runs

    def train_on_val_one_epoch(self, train_loader) -> float:
        """Train on validation set."""
        self.train()

        loss_all = 0.0
        n_runs = 0
        bce = nn.BCELoss()

        for batch in (pbar := tqdm.tqdm(train_loader)):
            data = batch['data'].to(self.device)
            target = batch['labels'].to(self.device).float()
            target_hat = torch.bernoulli(self.get_target(batch))

            self.optimizer.zero_grad()
            y_pred = self.model(data, target_hat, target)['y']
            loss = bce(y_pred, target)

            self.accelerator.backward(loss)

            loss_all += loss.item()
            n_runs += 1

            self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=10)
            self.optimizer.step()

            pbar.set_description(f'training loss on validation: {loss_all/n_runs:.4f}')

        return loss_all / n_runs

    @torch.inference_mode()
    def predict(self, batch: Dict, **kwargs) -> torch.Tensor:
        """Sample predictions from model."""
        data = batch['data'].to(self.device)
        target_dist = self.get_target(batch)

        y = self.model.sample(data, target_dist)
        return y.float()
