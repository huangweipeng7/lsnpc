import argparse
import logging
import numpy as np
import sys
import torch
import torchvision.transforms as transforms
from functools import partial

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
            
    Returns:
        dict: Dictionary containing:
            - 'train_dataset': Training dataset with transforms applied
            - 'val_dataset0': Validation dataset 0 with validation transforms
            - 'val_dataset1': Validation dataset 1 with training transforms 
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
    train_dataset.set_transform(
        partial(batch_transform, transform=train_transform)
    )
    
    val_dataset0 = data_class.val_data0.with_transform(
        partial(batch_transform, transform=val_transform)
    )
    val_dataset1 = data_class.val_data1.with_transform(
        partial(batch_transform, transform=train_transform)
    )
    test_dataset = data_class.test_data.with_transform(
        partial(batch_transform, transform=val_transform)
    )

    logger.info('Dataset loaded successfully!')
    n_labels = data_class.get_number_classes()

    return {
        'train_dataset': train_dataset, 
        'val_dataset0': val_dataset0,  
        'val_dataset1': val_dataset1, 
        'test_dataset': test_dataset,
        'n_labels': n_labels
    }