"""Base training pipeline components for consistent training workflows.

This module provides base classes and utilities for training pipelines,
reducing code duplication across different training scripts.
"""

import hashlib
import logging
import numpy as np
import torch
import orjson
from accelerate import Accelerator
from datetime import datetime
from torch.utils.data import DataLoader 
from typing import Any, Dict, Optional


class BaseTrainingArguments:
    """Base class for training arguments with common attributes."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert arguments to dictionary."""
        return {
            **vars(self),
        }


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def generate_run_uid(arg_dict: Dict[str, Any], run_index: int = 0) -> str:
    """Generate unique identifier for a training run.
    
    Args:
        arg_dict: Dictionary of training arguments.
        run_index: Index of the current run (for multi-run experiments).
        
    Returns:
        MD5 hash string as unique identifier.
    """
    arg_dict_copy = arg_dict.copy()
    arg_dict_copy['run_index'] = run_index
    return hashlib.md5(
        orjson.dumps(arg_dict_copy, option=orjson.OPT_SORT_KEYS)
    ).hexdigest()


def create_timestamp() -> str:
    """Create timestamp string for run tracking.
    
    Returns:
        Timestamp string in YYYYMMDD_HH_MM_SS format.
    """
    return datetime.now().strftime("%Y%m%d_%H_%M_%S")


class BaseTrainingPipeline:
    """Base class for all training pipelines.
    
    This class provides common functionality for training workflows including:
    - Seed management
    - Data loader creation
    - Accelerator setup
    - Run tracking utilities
    
    Subclasses should override:
    - build_model(): To define model architecture
    - get_loss_fn(): To define loss function
    - get_optimizer(): To define optimizer
    - train(): Main training logic
    """
    
    def __init__(
        self,
        model_args: Any,
        data_args: Any,
        train_args: Any,
        gradient_accumulation_steps: int = 1,
    ):
        """Initialize base training pipeline.
        
        Args:
            model_args: Model configuration arguments.
            data_args: Data configuration arguments.
            train_args: Training configuration arguments.
            gradient_accumulation_steps: Number of steps for gradient accumulation.
        """
        self.model_args = model_args
        self.data_args = data_args
        self.train_args = train_args
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        
    def setup_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        set_random_seeds(self.train_args.seed)
        
    def setup_logging(self) -> None:
        """Setup logging for the training pipeline.
        
        Configures console logging with optional file logging.
        Uses train_args attributes if available: logging_level, verbose, log_file.
        """
        # Get logging level from train_args, default to INFO
        level = getattr(self.train_args, 'logging_level', 'INFO')
        verbose = getattr(self.train_args, 'verbose', False)
        log_file = getattr(self.train_args, 'log_file', None)
        
        # Convert string level to logging constant
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        
        # Override with DEBUG if verbose
        if verbose:
            numeric_level = logging.DEBUG
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)
        
        # Remove existing handlers to avoid duplicates
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler (if log_file is specified)
        if log_file is not None:
            from pathlib import Path
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
    def load_dataset_module(self) -> Any:
        """Load appropriate dataset module based on dataset name.
        
        Returns:
            Dataset module for data loading.
        """
        from data_process import data_utils
        return data_utils
    
    def create_data_loaders(
        self,
        data: Dict[str, Any],
        batch_size_multiplier: float = 1.0,
        include_clean_loader: bool = False,
    ) -> Dict[str, Optional[DataLoader]]:
        """Create data loaders for training, validation, and testing.
        
        Args:
            data: Dictionary containing datasets.
            batch_size_multiplier: Multiplier for batch sizes.
            include_clean_loader: Whether to include clean validation loader.
            
        Returns:
            Dictionary of data loaders with keys:
            - 'train': Training loader
            - 'val': Validation loader (may be noisy)
            - 'clean': Clean validation loader (never noisified)
            - 'test': Test loader
        """
        train_dataset = data['train_dataset']
        val_dataset = data['val_dataset']
        clean_val_dataset = data.get('clean_val_dataset')
        test_dataset = data['test_dataset']
        
        batch_size = self.train_args.batch_size
        num_workers = self.data_args.num_workers
        
        loaders = {
            'train': DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                drop_last=True,
                shuffle=True,
                pin_memory=True,
            ),
            'val': DataLoader(
                dataset=val_dataset,
                batch_size=int(batch_size * 2 * batch_size_multiplier),
                num_workers=num_workers,
                drop_last=True,
                shuffle=False,
                pin_memory=True,
            ),
            'test': DataLoader(
                dataset=test_dataset,
                batch_size=int(batch_size * 4 * batch_size_multiplier),
                num_workers=num_workers,
                drop_last=False,
                shuffle=False,
                pin_memory=True,
            ),
        }
        
        # Add clean validation loader
        if include_clean_loader and clean_val_dataset is not None:
            loaders['clean'] = DataLoader(
                dataset=clean_val_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                drop_last=True,
                shuffle=True,
                pin_memory=True,
            )
        else:
            loaders['clean'] = None
            
        return loaders
    
    def prepare_with_accelerator(self, *components):
        """Prepare components with accelerator for distributed training.
        
        Args:
            *components: PyTorch modules and data loaders to prepare.
            
        Returns:
            Prepared components in same order as input.
        """
        return self.accelerator.prepare(*components)
    
    def wait_for_processes(self) -> None:
        """Synchronize processes in distributed training."""
        self.accelerator.wait_for_everyone()
        
    def build_model(self, *args, **kwargs) -> torch.nn.Module:
        """Build the model architecture.
        
        This method should be overridden by subclasses.
        
        Raises:
            NotImplementedError: If not overridden by subclass.
        """
        raise NotImplementedError("Subclasses must implement build_model()")
    
    def get_loss_fn(self, *args, **kwargs) -> torch.nn.Module:
        """Get the loss function.
        
        This method should be overridden by subclasses.
        
        Raises:
            NotImplementedError: If not overridden by subclass.
        """
        raise NotImplementedError("Subclasses must implement get_loss_fn()")
    
    def get_optimizer(
        self,
        model: torch.nn.Module,
    ) -> torch.optim.Optimizer:
        """Get optimizer for the model.
        
        Default implementation uses AdamW. Override for custom optimizers.
        
        Args:
            model: PyTorch model to optimize.
            
        Returns:
            Optimizer instance.
        """
        return torch.optim.AdamW(
            model.parameters(),
            lr=self.train_args.lr, 
            weight_decay=self.train_args.weight_decay,
            fused=True
        )
    
    def get_lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Get learning rate scheduler.
        
        Default implementation uses CosineAnnealingLR. Override for custom schedulers.
        
        Args:
            optimizer: Optimizer to schedule.
            
        Returns:
            Learning rate scheduler or None.
        """
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.train_args.n_train_epoch,
        )
    
    def create_trainer(self, model, optimizer, loss_fn, lr_scheduler, loaders):
        """Create trainer instance.
        
        This method should be overridden by subclasses to return the appropriate
        trainer class.
        
        Args:
            model: Prepared model
            optimizer: Prepared optimizer
            loss_fn: Loss function
            lr_scheduler: Learning rate scheduler
            loaders: Dictionary of data loaders
            
        Returns:
            Trainer instance
        """
        raise NotImplementedError("Subclasses must implement create_trainer()")
    
    def run_training_loop(
        self,
        run_index: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        loaders: Dict[str, Optional[DataLoader]],
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Execute standard training loop.
        
        Args:
            run_index: Current run index.
            model: Model to train.
            optimizer: Optimizer.
            loss_fn: Loss function.
            lr_scheduler: Learning rate scheduler.
            loaders: Dictionary of data loaders.
            extra_args: Additional arguments for run tracking.
        """
        # Generate run UID
        arg_dict = {
            **self.model_args.__dict__,
            **self.data_args.__dict__,
            **self.train_args.__dict__,
        }
        if extra_args:
            arg_dict.update(extra_args)
            
        uid = generate_run_uid(arg_dict, run_index)
        arg_dict['uid'] = uid
        arg_dict['time'] = create_timestamp()
        arg_dict['run_index'] = run_index
        
        # Prepare with accelerator
        prepared_components = self.prepare_with_accelerator(
            model,
            optimizer,
            loss_fn,
            lr_scheduler if lr_scheduler is not None else optimizer,
            loaders['train'],
            loaders['val'],
            loaders['clean'] if loaders['clean'] is not None else optimizer,
            loaders['test'],
        )
         
        # Unpack prepared components
        model, optimizer, loss_fn, _, train_loader, val_loader, clean_loader, test_loader = prepared_components
        
        # Create trainer
        trainer = self.create_trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            loaders={
                'train': train_loader,
                'val': val_loader,
                'clean': clean_loader,
                'test': test_loader,
            },
            arg_dict=arg_dict,
        ) 
        
        # Train
        gc_interval = getattr(self.train_args, 'gc_interval', 5)
        trainer.train_model(
            n_epochs=self.train_args.n_train_epoch,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            verbose=True,
            clean_set_loader=clean_loader,
            gc_interval=gc_interval,
        )

        # Synchronize
        self.wait_for_processes()
        print(f'Round {run_index} finished.\n\n')


class VAETrainingPipeline(BaseTrainingPipeline):
    """Base class for VAE-based training pipelines.
    
    Provides additional utilities specific to VAE models including:
    - Beta-VAE parameters
    - Gradient clipping
    - Semi-supervised learning support
    """
    
    def __init__(
        self,
        model_args: Any,
        data_args: Any,
        train_args: Any,
        gradient_accumulation_steps: int = 1,
        n_labels: Optional[int] = None,
    ):
        super().__init__(model_args, data_args, train_args, gradient_accumulation_steps)
        
        # Check for VAE-specific arguments
        self.beta = getattr(train_args, 'beta', 0.0001)
        self.grad_norm = getattr(train_args, 'grad_norm', 2)
        self.semi_sup = getattr(train_args, 'semi_sup', False)
        self.is_ablation = getattr(train_args, 'is_ablation', False)
        
        # Cache number of labels to avoid reloading data
        self._n_labels = n_labels
        
    def get_optimizer(
        self,
        model: torch.nn.Module,
    ) -> torch.optim.Optimizer:
        """Get optimizer with VAE-specific settings."""
        return torch.optim.AdamW(
            model.parameters(),
            lr=self.train_args.lr,
            weight_decay=self.train_args.weight_decay,
            fused=True,
        )
    
    def create_trainer(self, model, optimizer, loss_fn, lr_scheduler, loaders, arg_dict):
        """Create VAE trainer instance.
        
        Args:
            model: Prepared model
            optimizer: Prepared optimizer
            loss_fn: Loss function
            lr_scheduler: Learning rate scheduler
            loaders: Dictionary of data loaders
            arg_dict: Argument dictionary for tracking
            
        Returns:
            VAETrainer instance
        """
        from trainers import VAETrainer
        
        return VAETrainer(
            model=model,
            n_labels=self._get_n_labels(),
            loss_fn=loss_fn,
            optimizer=optimizer,
            arg_dict=arg_dict,
            lr_scheduler=lr_scheduler,
            train_on_val=self.semi_sup,
            grad_norm=self.grad_norm,
            eval_test_at_final_loop_only=getattr(
                self.train_args, 'eval_test_at_final_loop_only', False
            ),
            metric_storing_path=f"./results/{arg_dict['dataset']}_results.csv",
            accelerator=self.accelerator,
        )
    
    def _get_n_labels(self) -> int:
        """Get number of labels from data.
        
        Returns cached value if available to avoid reloading data.
        """
        if self._n_labels is not None:
            return self._n_labels
            
        # Fallback: load data (for backward compatibility)
        data_utils = self.load_dataset_module()
        data = data_utils.load_data(self.data_args)
        return data['n_labels']


class StandardTrainingPipeline(BaseTrainingPipeline):
    """Base class for standard (non-VAE) training pipelines.
    
    Provides utilities for standard classifier training.
    """
    
    def __init__(
        self,
        model_args: Any,
        data_args: Any,
        train_args: Any,
        gradient_accumulation_steps: int = 1,
        n_labels: Optional[int] = None,
    ):
        super().__init__(model_args, data_args, train_args, gradient_accumulation_steps)
        
        # Cache number of labels to avoid reloading data
        self._n_labels = n_labels
    
    def create_trainer(self, model, optimizer, loss_fn, lr_scheduler, loaders, arg_dict):
        """Create standard trainer instance.
        
        Args:
            model: Prepared model
            optimizer: Prepared optimizer
            loss_fn: Loss function
            lr_scheduler: Learning rate scheduler
            loaders: Dictionary of data loaders
            arg_dict: Argument dictionary for tracking
            
        Returns:
            Trainer instance
        """
        from trainers import Trainer
        
        return Trainer(
            model=model,
            n_labels=self._get_n_labels(),
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            arg_dict=arg_dict,
            eval_test_at_final_loop_only=getattr(
                self.train_args, 'eval_test_at_final_loop_only', False
            ),
            metric_storing_path=f"./results/{arg_dict['dataset']}_results.csv",
            accelerator=self.accelerator,
        )
    
    def _get_n_labels(self) -> int:
        """Get number of labels from data.
        
        Returns cached value if available to avoid reloading data.
        """
        if self._n_labels is not None:
            return self._n_labels
            
        # Fallback: load data (for backward compatibility)
        data_utils = self.load_dataset_module()
        data = data_utils.load_data(self.data_args)
        return data['n_labels']
