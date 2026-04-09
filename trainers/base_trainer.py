"""Base trainer class for multi-label classification models."""

import gc
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import orjson
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader

from metrics import test
from utils import store_results


# Global defaults
DEFAULT_GRAD_NORM = 2.0
DEFAULT_GC_INTERVAL = 5  # Run garbage collection every N epochs

logger = logging.getLogger(__name__)


def log_metric(data_type: str, batch: Dict[str, float]) -> None:
    """Log training/validation metrics."""
    logger.info(
        f"{data_type} loss: {batch['loss']:.4f}, "
        f"macro f1: {batch['macro_f1']:.4f}, "
        f"micro f1: {batch['micro_f1']:.4f}, "
        f"mAP: {batch['mAP']:.4f}"
    )


class Trainer:
    """Base trainer class for multi-label classification models.

    Args:
        n_labels: Number of labels.
        model: PyTorch model to train.
        loss_fn: Loss function.
        optimizer: PyTorch optimizer.
        arg_dict: Dictionary of arguments and hyperparameters.
        lr_scheduler: Learning rate scheduler. Default: None.
        train_on_val: Whether to train on validation set. Default: False.
        eval_test_at_final_loop_only: Whether to evaluate test only at final epoch. Default: True.
        metric_storing_path: Path to store results CSV. Default: './runs/results.csv'.
        accelerator: Accelerator for distributed training. Default: None.
    """

    def __init__(
        self,
        n_labels: int,
        model: nn.Module,
        loss_fn,
        optimizer: torch.optim.Optimizer,
        arg_dict: Dict,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        train_on_val: bool = False,
        eval_test_at_final_loop_only: bool = True,
        metric_storing_path: Union[str, Path] = './runs/results.csv',
        accelerator: Accelerator = None,
        grad_norm: float = DEFAULT_GRAD_NORM,
        gradient_accumulation_steps: int = 1,
        zero_grad_on_step: bool = True
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.device = accelerator.device if accelerator else 'cpu'
        self.grad_norm = grad_norm
        
        # Gradient accumulation support
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.zero_grad_on_step = zero_grad_on_step
        
        # Reuse BCEWithLogitsLoss for evaluation
        self._bce_loss = nn.BCEWithLogitsLoss()

        # Best model selection vs early stopping (mutually exclusive options)
        # Option 1: save_best_only - Only save when validation improves (default: True)
        self.save_best_only = arg_dict.get('save_best_only', True)
        
        # Option 2: early_stopping_patience - Stop training after N epochs without improvement (default: 0 = disabled)
        self.early_stopping_patience = arg_dict.get('early_stopping_patience', 0)
        self.early_stopping_min_delta = arg_dict.get('early_stopping_min_delta', 0.0)
        self.epochs_without_improvement = 0

        self.uid = arg_dict['uid']
        self.arg_dict = arg_dict
        self.train_on_val = train_on_val
        self.n_labels = n_labels

        self.eval_test_at_final_loop_only = eval_test_at_final_loop_only

        self.metric_storing_path = Path(metric_storing_path)
        self.metric_storing_path.parent.mkdir(parents=True, exist_ok=True)

        self.res_path = Path(self.arg_dict['result_dir']) / (
            f'./{self.arg_dict["dataset"]}_{self.arg_dict["noise_type"]}_'
            f'{self.arg_dict["noise_rate"]}_{self.arg_dict["img_encoder"]}_'
            f'ep{self.arg_dict["n_train_epoch"]}_rd{self.arg_dict["run_index"]}/'
        )

        self.metric = 0.0
        self.best_ep = 0

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
        """Main training loop.

        Args:
            n_epochs: Number of training epochs.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            test_loader: Test data loader.
            verbose: Whether to log detailed metrics.
            clean_set_loader: Clean validation set for training on val.
            gc_interval: Run garbage collection every N epochs.
        """
        logger.info(f'Training with uid: {self.uid}')

        for epoch in range(n_epochs):
            logger.info(f'Epoch {epoch}')
            train_loss = self.train_one_epoch(train_loader, epoch=epoch)

            if self.train_on_val:
                assert clean_set_loader is not None
                self.train_on_val_one_epoch(clean_set_loader)

            if self.accelerator.is_main_process:
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

    @torch.inference_mode()
    def eval_and_save(
        self,
        epoch: int,
        n_epochs: int,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        verbose: bool = False
    ) -> None:
        """Evaluate model and save checkpoints."""
        self.eval()

        if val_loader is not None:
            v_batch = test(self, val_loader, self._bce_loss)

            if verbose:
                log_metric('val', v_batch)

            v = v_batch['micro_f1']
            
            # Determine which mode is active (mutually exclusive)
            use_early_stopping = self.early_stopping_patience > 0
            use_best_model_saving = self.save_best_only and not use_early_stopping
            
            # Option 1: Best model selection only (when early stopping is disabled)
            if use_best_model_saving and epoch > n_epochs // 2:
                if v >= self.metric:
                    self.metric = v
                    self.best_ep = epoch
                    self.save_model(self.arg_dict, self.res_path)
            
            # Option 2: Early stopping (stops training when no improvement for N epochs)
            if use_early_stopping:
                if v > self.metric + self.early_stopping_min_delta:
                    self.epochs_without_improvement = 0
                    self.metric = v
                    self.best_ep = epoch
                    self.save_model(self.arg_dict, self.res_path)  # Save best on improvement
                else:
                    self.epochs_without_improvement += 1
                    if self.epochs_without_improvement >= self.early_stopping_patience:
                        logger.info(
                            f'Early stopping triggered after {epoch + 1} epochs. '
                            f'No improvement for {self.early_stopping_patience} consecutive epochs.'
                        )
                        return  # Early stop - don't continue to test evaluation

        if test_loader is not None:
            if not self.eval_test_at_final_loop_only:
                t_batch = test(self, test_loader, self._bce_loss)
            elif epoch == n_epochs - 1:
                try:
                    self.model.load_state_dict(
                        torch.load(
                            self.res_path / f'{self.uid}.pth', 
                            weights_only=True
                        )
                    )
                except FileNotFoundError:
                    logger.warning('No checkpoint found, using the last one')

                t_batch = test(self, test_loader, self._bce_loss)
            else:
                return

            if verbose:
                log_metric('test', t_batch)
                logger.info(f'best epoch: {self.best_ep}')

            store_results(
                {**t_batch, **self.arg_dict, 'epoch': epoch, 'data_split': 'test'},
                str(self.metric_storing_path)
            )
            logger.info("Finish saving results")

    def train_one_epoch(
        self,
        train_loader: DataLoader,
        epoch: int = -1
    ) -> float:
        """Train for one epoch."""
        self.train()

        loss_all = 0.0
        n_runs = 0

        for batch in (pbar := tqdm.tqdm(train_loader)):
            with self.accelerator.accumulate(self.model):
                data = batch['data']
                target = batch['labels'].float()

                self.optimizer.zero_grad()
                pred = self.model(data)
                loss = self.loss_fn(pred, target)

                self.accelerator.backward(loss)

                loss_all += loss.item()
                n_runs += 1

                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_norm
                )
                self.optimizer.step()

            pbar.set_description(f'training loss: {loss_all/n_runs:.4f}')

        return loss_all / n_runs

    def train_on_val_one_epoch(self, train_loader: DataLoader, epoch: int = -1) -> float:
        """Train on validation set for one epoch.
        
        Override this method in subclasses that support training on validation set.
        Default implementation raises NotImplementedError.
        """
        raise NotImplementedError(
            "train_on_val_one_epoch is not implemented in base Trainer class. "
            "Use VAETrainer or another subclass that implements this method."
        )
    
    @torch.inference_mode()
    def predict(self, batch) -> torch.Tensor:
        """Make predictions without gradients."""
        data = batch['data']
        return F.sigmoid(self.model(data))

    def eval(self) -> None:
        """Set model to evaluation mode."""
        self.model.eval()

    def train(self) -> None:
        """Set model to training mode."""
        self.model.train()

    def save_model(self, arg_dict: Dict, path: Union[str, Path]) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        model = self.accelerator.unwrap_model(self.model)
        torch.save(model.state_dict(), path / f'{self.uid}.pth')

        with open(path / f'{self.uid}.json', 'wb') as f:
            f.write(orjson.dumps(arg_dict, option=orjson.OPT_INDENT_2))

    def load_model(self, path: Union[str, Path]) -> None:
        """Load model checkpoint from path.""" 
        self.model.load_state_dict(
            torch.load(Path(path) / f'{self.uid}.pth', weights_only=True)
        )
        logger.info(f'Loaded model from {path}')

    def reset_early_stopping(self) -> None:
        """Reset early stopping counter to allow continued training."""
        self.epochs_without_improvement = 0
        self.metric = 0.0
        logger.info('Early stopping counters reset')
