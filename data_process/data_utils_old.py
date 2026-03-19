"""Legacy data loading utilities for memory-efficient dataset handling.

This module provides backward-compatible data loading functions for datasets
that require memory-efficient processing (loading data incrementally rather
than all at once).

Note: This is a legacy module kept for compatibility. New projects should use
the modern data_utils.py instead.
"""

import logging
from typing import Dict, Any

import numpy as np
import torch
import torchvision.transforms as transforms

from .coco_old import COCO2014
from utils import MultiScaleCrop, Warp

logger = logging.getLogger(__name__)


def load_data(args: Any) -> Dict[str, Any]:
    """Load datasets with memory-efficient processing.
    
    This function loads training, validation, and test datasets using
    incremental loading to handle large datasets that don't fit in memory.
    
    Args:
        args: Arguments containing dataset configuration:
            - root: Root directory for dataset
            - image_size: Target image size
            - noise_type: Type of noise injection ('symmetric', 'pairflip')
            - noise_rate: Noise injection rate (0.0 to 1.0)
            - split_percentage: Train/validation split ratio
            
    Returns:
        Dictionary containing:
            - train_dataset: Training dataset
            - val_dataset0: First validation split
            - val_dataset1: Second validation split  
            - test_dataset: Test dataset
            - n_labels: Number of labels (80 for COCO)
            
    Raises:
        ValueError: If unsupported dataset is requested.
    """
    print(f'Dataset root: {args.root}')
    
    # Set random seeds for reproducibility
    seed = 1000
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Define data transformations
    train_transform = transforms.Compose([
        MultiScaleCrop(
            args.image_size, 
            scales=(1.0, 0.875, 0.75, 0.66, 0.5), 
            max_distort=2
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        Warp(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load datasets based on type
    if args.dataset == 'coco':
        train_dataset = COCO2014(
            args.root,
            phase='train',
            noise_type=args.noise_type,
            noise_rate=args.noise_rate,
            transform=train_transform,
            split_per=args.split_percentage,
            random_seed=seed
        )

        val_dataset0 = COCO2014(
            args.root,
            phase='val0',
            noise_type=args.noise_type,
            noise_rate=args.noise_rate,
            transform=val_transform,
            split_per=args.split_percentage,
            random_seed=seed
        )

        val_dataset1 = COCO2014(
            args.root,
            phase='val1',
            noise_type=args.noise_type,
            noise_rate=args.noise_rate,
            transform=train_transform,
            split_per=args.split_percentage,
            random_seed=seed
        )

        test_dataset = COCO2014(
            args.root,
            phase='test',
            transform=val_transform
        )
    else:
        raise ValueError(
            f"Unsupported dataset: '{args.dataset}'. "
            "Only 'coco' is supported in this memory-efficient version."
        )

    print('Dataset loading complete!')
    n_labels = 80  # COCO has 80 classes

    return {
        'train_dataset': train_dataset,
        'val_dataset0': val_dataset0,
        'val_dataset1': val_dataset1,
        'test_dataset': test_dataset,
        'n_labels': n_labels
    }