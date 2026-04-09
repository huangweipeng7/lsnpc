"""Training utilities for model loading, data utilities, and common operations."""

import torch
import torch.nn as nn
from modelscope import ViTModel
from torchvision.models import resnet50, ResNet50_Weights
from transformers import LevitModel 
from typing import Tuple, Any, Dict

import dnn.hlc as hlc 
from dnn.mlc import ViTModelWrapper
from dnn.mlc import MultilabelClassifier
from dnn.mcm import MCMClassifier


def get_encoder(img_encoder: str) -> Tuple[nn.Module, int]:
    """Load image encoder architecture.
    
    Args:
        img_encoder: Encoder name ('resnet50', 'levit', or 'vit224').
        
    Returns:
        Tuple of (encoder module, embedding size).
        
    Raises:
        AttributeError: If encoder name is not recognized.
    """
    if img_encoder == 'resnet50':
        encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
        encoder = torch.nn.Sequential(*(list(encoder.children())[:-1]))
        encoder.fc = nn.Flatten()
        emb_size = 2048
    elif img_encoder == 'levit':
        encoder = ViTModelWrapper(
            LevitModel.from_pretrained(
                'local_models/levit', local_files_only=True
            )
        )
        emb_size = 384
    elif img_encoder == 'vit224':
        encoder = ViTModelWrapper(
            ViTModel.from_pretrained(
                'AI-ModelScope/vit-base-patch16-224', 
            )
        )
        emb_size = 768
    else:
        raise AttributeError('Image feature encoder is not defined...')
    
    return encoder, emb_size


def import_data_utils(dataset: str) -> Any:
    """Import appropriate dataset utility module based on dataset name.

    Args:
        dataset: Name of the dataset.

    Returns:
        Dataset utility module for data loading.
    """
    from data_process import data_utils
    return data_utils


def get_pretrained_model(
    clf_name: str, 
    pretrained_clf_name: str, 
    encoder: nn.Module, 
    emb_size: int, 
    n_labels: int
) -> nn.Module:
    """Load pretrained classifier model.
    
    Args:
        clf_name: Classifier type ('mlclf', 'asl', 'addgcn', 'hlc', or 'mcm').
        pretrained_clf_name: Path to pretrained weights file.
        encoder: Encoder architecture.
        emb_size: Embedding size.
        n_labels: Number of labels.
        
    Returns:
        Pretrained classifier with frozen parameters.
        
    Raises:
        AttributeError: If classifier name is not recognized.
    """
    if clf_name in ['mlclf', 'asl']: 
        pretrained_clf = MultilabelClassifier(encoder, emb_size, n_labels)
    elif clf_name == 'addgcn':
        pretrained_clf = hlc.get_model(encoder, emb_size, n_labels)
    elif clf_name == 'hlc':
        pretrained_clf = hlc.get_model(encoder, emb_size, n_labels)
    elif clf_name == 'mcm':
        pretrained_clf = MCMClassifier(encoder, emb_size, n_labels)
    else:
        raise AttributeError('Not recognized classifier')

    pretrained_clf.load_state_dict(
        torch.load(pretrained_clf_name, weights_only=True)
    )
    
    # Freeze parameters
    for p in pretrained_clf.parameters():
        p.requires_grad = False

    return pretrained_clf


def load_dataset_module(dataset_name: str) -> Any:
    """Load dataset module based on dataset name.
    
    This is a convenience wrapper around import_data_utils with clearer naming.
    
    Args:
        dataset_name: Name of the dataset.
        
    Returns:
        Dataset utility module.
    """
    return import_data_utils(dataset_name)


def verify_data_consistency(
    data: Dict[str, Any],
    verbose: bool = True
) -> None:
    """Verify consistency of loaded datasets.
    
    Args:
        data: Dictionary containing datasets.
        verbose: Whether to print diagnostic information.
    """
    if not verbose:
        return
        
    val_dataset = data['val_dataset']
    test_dataset = data['test_dataset']
    
    print('Validation dataset true labels:', val_dataset.true_labels[:10])
    print('Validation dataset labels:', val_dataset.labels[:10])
    
    if hasattr(val_dataset, 'true_labels') and hasattr(test_dataset, 'true_labels'):
        print('Test dataset true labels:', test_dataset.true_labels[:10])
    
    print('Test dataset:', test_dataset)


def create_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    optimizer_type: str = 'adamw'
) -> torch.optim.Optimizer:
    """Create optimizer for model.
    
    Args:
        model: Model to optimize.
        lr: Learning rate.
        weight_decay: Weight decay factor.
        optimizer_type: Optimizer type ('adam', 'adamw', or 'sgd').
        
    Returns:
        Optimizer instance.
    """
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_type == 'adamw':
        return GrokOptimizer(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_type == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    n_epochs: int,
    **kwargs
) -> Any:
    """Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule.
        scheduler_type: Scheduler type ('cosine', 'step', 'multistep', etc.).
        n_epochs: Number of training epochs.
        **kwargs: Additional scheduler-specific arguments.
        
    Returns:
        Learning rate scheduler or None.
    """
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == 'cosine':
        eta_min = kwargs.get('eta_min', 0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=n_epochs,
            eta_min=eta_min,
        )
    elif scheduler_type == 'step':
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    elif scheduler_type == 'multistep':
        milestones = kwargs.get('milestones', [30, 60, 90])
        gamma = kwargs.get('gamma', 0.1)
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
        )
    elif scheduler_type == 'none' or scheduler_type == '':
        return None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
