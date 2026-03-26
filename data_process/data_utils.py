"""Optimized data utilities for multi-label classification with caching and lazy loading.

This module provides utilities for loading and processing datasets with support for:
- Disk-based caching to speed up repeated loading
- Lazy image loading for memory efficiency
- Multi-worker DataLoader for parallel processing
- Fast image decoding with pillow-simd support
"""

import logging
from pathlib import Path
import pickle
from functools import partial
from typing import Dict, Union

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from .deepfashion import DeepFashion
from .coco import COCO2014
from .tomato import Tomato
from .voc import Voc2007, Voc2012
from .nuswide import NUSWide
from utils import MultiScaleCrop, Warp

logger = logging.getLogger(__name__)


def batch_transform(batch, transform):
    """Apply transformation to a batch of images and labels.
    
    Args:
        batch (dict): Dictionary containing 'data' (images) and 'labels'.
        transform (callable): Transformation pipeline to apply to images.
        
    Returns:
        dict: Transformed batch with images and labels as tensors.
    """
    batch['data'] = [transform(x) for x in batch['data']]
    batch['labels'] = [torch.tensor(x) for x in batch['labels']]
    return batch


def batch_transform_fast(batch, transform):
    """Optimized batch transformation using list comprehension.
    
    This version is optimized for speed by minimizing function call overhead.
    
    Args:
        batch (dict): Dictionary containing 'data' (images) and 'labels'.
        transform (callable): Transformation pipeline to apply to images.
        
    Returns:
        dict: Transformed batch with images and labels as tensors.
    """
    # Use list comprehension for faster processing
    batch['data'] = [transform(img) for img in batch['data']]
    batch['labels'] = torch.stack([torch.tensor(label) for label in batch['labels']])
    return batch


def get_cache_path(dataset_name: str, cache_dir: Union[str, Path] = './data_cache') -> Path:
    """Generate cache file path for dataset.
    
    Args:
        dataset_name: Name of the dataset
        cache_dir: Directory to store cache files
        
    Returns:
        Path to cache file
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f'{dataset_name}_cache.pkl'
    return cache_file


def load_data(args):
    """Load and prepare dataset based on configuration.
    
    Args:
        args (argparse.Namespace): Configuration object with the following attributes:
            - root (str): Root directory of the dataset
            - image_size (int): Size to resize images to
            - split_percentage (float): Proportion of data for training/validation split
            - noise_type (str): Type of noise to apply ['pairflip', 'symmetric', 'asymmetric']
            - noise_rate (float): Noise rate (0.0 to 1.0)
            - dataset (str): Dataset name ['coco', 'voc2007', 'voc2012', 'tomato', 'deepfashion', 'nuswide']
            - num_workers (int): Number of workers for data loading
            - noisy_val (bool): Whether to add noise to validation data
            - use_cache (bool): Whether to use dataset caching (default: True)
            
    Returns:
        dict: Dictionary containing:
            - 'train_dataset': Training dataset with transforms applied
            - 'val_dataset': Validation dataset (may be noisy based on noisy_val parameter)
            - 'clean_val_dataset': Clean validation dataset (never noisified)
            - 'test_dataset': Test dataset with validation transforms
            - 'num_classes': Number of classes in the dataset
    
    Raises:
        ValueError: If an unsupported dataset name is provided
    """
    # Define image transformations for training and validation
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
    
    # Map dataset names to their corresponding classes
    dataset_map = {
        'coco': COCO2014,
        'voc2007': Voc2007,
        'voc2012': Voc2012,
        'tomato': Tomato,
        'deepfashion': DeepFashion,
        'nuswide': NUSWide
    }
    
    # Validate dataset argument
    if args.dataset not in dataset_map:
        available_datasets = ', '.join(dataset_map.keys())
        raise ValueError(f"Unsupported dataset: '{args.dataset}'. "
                         f"Available datasets: [{available_datasets}]")
    
    # Check for cached dataset
    use_cache = getattr(args, 'use_cache', True)
    cache_dir = getattr(args, 'cache_dir', './data_cache')
    cache_file = get_cache_path(f"{args.dataset}_{args.noise_type}_{args.noise_rate}", cache_dir)
    
    if use_cache and cache_file.exists():
        logger.info(f'Loading dataset from cache: {cache_file}')
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        
        # Apply transforms to cached datasets
        train_dataset = cached_data['train_data'].with_format(
            'torch', columns=['labels'], output_all_columns=True
        )
        train_dataset.set_transform(partial(batch_transform_fast, transform=train_transform))
        
        # Handle both old (val_data0/val_data1) and new (val_data/clean_val_data) cache formats
        if 'val_data' in cached_data:
            val_dataset = cached_data['val_data'].with_transform(
                partial(batch_transform_fast, transform=val_transform)
            )
            clean_val_dataset = cached_data['clean_val_data'].with_transform(
                partial(batch_transform_fast, transform=train_transform)
            )
        else:
            # Backward compatibility: old cache used val_data0/val_data1
            val_dataset = cached_data['val_data0'].with_transform(
                partial(batch_transform_fast, transform=val_transform)
            )
            clean_val_dataset = cached_data['val_data1'].with_transform(
                partial(batch_transform_fast, transform=train_transform)
            )
        
        test_dataset = cached_data['test_data'].with_transform(
            partial(batch_transform_fast, transform=val_transform)
        )
        
        n_labels = cached_data['n_labels']
        logger.info('Dataset loaded from cache successfully!')
        
    else:
        # Initialize the appropriate dataset class
        DataClass = dataset_map[args.dataset]
        
        try:
            data_class = DataClass(
                args.root, 
                noise_type=args.noise_type, 
                noise_rate=args.noise_rate,  
                split_per=args.split_percentage,
                num_workers=args.num_workers,
                noisy_val=args.noisy_val
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize dataset '{args.dataset}' "
                               f"with root directory '{args.root}': {str(e)}")

        # Format datasets with proper transforms
        train_dataset = data_class.train_data.with_format(
            'torch', columns=['labels'], output_all_columns=True
        )
        train_dataset.set_transform(partial(batch_transform_fast, transform=train_transform))
        
        val_dataset = data_class.val_data.with_transform(
            partial(batch_transform_fast, transform=val_transform)
        )
        clean_val_dataset = data_class.clean_val_data.with_transform(
            partial(batch_transform_fast, transform=train_transform)
        )
        test_dataset = data_class.test_data.with_transform(
            partial(batch_transform_fast, transform=val_transform)
        )

        n_labels = data_class.get_number_classes()
        logger.info('Dataset loaded successfully!')
        
        # Save to cache
        if use_cache:
            logger.info(f'Saving dataset to cache: {cache_file}')
            cache_data = {
                'train_data': data_class.train_data,
                'val_data': data_class.val_data,
                'clean_val_data': data_class.clean_val_data,
                'test_data': data_class.test_data,
                'n_labels': n_labels
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)

    return {
        'train_dataset': train_dataset, 
        'val_dataset': val_dataset,  
        'clean_val_dataset': clean_val_dataset, 
        'test_dataset': test_dataset,
        'n_labels': n_labels
    }


def create_dataloader(
    dataset, 
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True
) -> DataLoader:
    """Create an optimized DataLoader from a datasets.Dataset.
    
    Args:
        dataset: HuggingFace datasets.Dataset or similar
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        
    Returns:
        torch.utils.data.DataLoader instance
    """
    def collate_fn(batch):
        """Custom collate function for efficient batching."""
        # Stack images and labels efficiently
        images = torch.stack([item['data'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        return {'data': images, 'labels': labels}
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=collate_fn
    )
    
    return dataloader


def load_data_with_dataloaders(args) -> Dict:
    """Load dataset and return DataLoaders for training.
    
    This function provides a complete data loading solution with optimized
    DataLoaders for training, validation, and testing.
    
    Args:
        args: Configuration object with dataset parameters
        
    Returns:
        dict: Dictionary containing:
            - 'train_loader': Training DataLoader
            - 'val_loader0': Validation DataLoader 0
            - 'val_loader1': Validation DataLoader 1
            - 'test_loader': Test DataLoader
            - 'num_classes': Number of classes
    """
    # Load datasets
    data = load_data(args)
    
    # Create DataLoaders
    train_loader = create_dataloader(
        data['train_dataset'],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    val_loader = create_dataloader(
        data['val_dataset'],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    clean_val_loader = create_dataloader(
        data['clean_val_dataset'],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    test_loader = create_dataloader(
        data['test_dataset'],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'clean_val_loader': clean_val_loader,
        'test_loader': test_loader,
        'num_classes': data['n_labels']
    }


def clear_cache(cache_dir: Union[str, Path] = './data_cache'):
    """Clear all cached datasets.

    Args:
        cache_dir: Directory containing cache files
    """
    cache_dir = Path(cache_dir)
    if cache_dir.exists():
        for file in cache_dir.glob('*.pkl'):
            file.unlink()
        logger.info(f'Cleared cache in {cache_dir}')
    else:
        logger.info(f'Cache directory {cache_dir} does not exist')
