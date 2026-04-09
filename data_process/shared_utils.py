"""Shared utilities for multi-label classification datasets.

This module contains common functions used across different dataset loaders,
including noise generation, dataset splitting, image loading, and caching utilities.
"""

import logging 
import numpy as np
from functools import partial
from numpy.testing import assert_array_almost_equal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union 
from PIL import Image, ImageFile

# Default random seed for reproducibility across the project
DEFAULT_RANDOM_SEED = 256

logger = logging.getLogger(__name__)

# Enable loading of truncated images and large images
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def load_image(
    path: Union[str, Path],
    convert_rgb: bool = True,
    fallback_size: Tuple[int, int] = (224, 224)
) -> 'Image.Image':
    """Load a single image with optimized decoding.

    This is a unified function for loading images across all datasets.

    Args:
        path: Full path to the image file
        convert_rgb: Whether to convert image to RGB mode
        fallback_size: Size of fallback image if loading fails

    Returns:
        PIL Image in RGB format (or fallback)
    """
    try:
        img = Image.open(path) 
        if convert_rgb and img.mode != 'RGB':
            img = img.convert('RGB') 

    except Exception as e:
        logger.warning(f"Failed to load image {path}: {e}")
        return Image.new('RGB', fallback_size, color=(128, 128, 128))

    return img


def load_image_from_parts(
    base_path: Union[str, Path],
    filename: str,
    extension: str = '.jpg',
    fallback_size: Tuple[int, int] = (224, 224)
) -> 'Image.Image':
    """Load image from base path and filename components.

    Args:
        base_path: Base directory path
        filename: Image filename without extension
        extension: File extension (default: '.jpg')
        fallback_size: Size of fallback image if loading fails

    Returns:
        PIL Image in RGB format
    """
    full_path = Path(base_path) / f'{filename}{extension}'
    return load_image(full_path, fallback_size=fallback_size)


def load_image_batch_lazy(
    batch: Dict[str, Any],
    path_key: str,
    base_path: Optional[Union[str, Path]] = None,
    extension: str = '.jpg',
    fallback_size: Tuple[int, int] = (224, 224)
) -> Dict[str, List['Image.Image']]:
    """Load a batch of images lazily.

    Helper function for use with datasets.Dataset.map()

    Args:
        batch: Batch dictionary from dataset
        path_key: Key in batch containing image path/filename
        base_path: Base directory path (optional)
        extension: File extension to append
        fallback_size: Size of fallback image if loading fails

    Returns:
        Dictionary with 'data' key containing list of images
    """
    images = []

    for path_or_name in batch[path_key]:
        if base_path:
            full_path = Path(base_path) / f'{path_or_name}{extension}'
        else:
            full_path = Path(f'{path_or_name}{extension}')

        img = load_image(full_path, fallback_size=fallback_size)
        images.append(img)

    return {'data': images}


def create_lazy_dataset(
    dataset,
    path_key: str = 'image_file',
    base_path: Optional[Union[str, Path]] = None,
    image_column: str = 'image_path',
    num_workers: int = 4,
    batch_size: int = 128,
    extension: str = '.jpg',
    remove_columns: Optional[List[str]] = None
):
    """Create a lazy dataset where images are loaded on-demand.

    This function creates a dataset that loads images only when accessed,
    reducing initial load time and memory usage.

    Args:
        dataset: datasets.Dataset with image paths and labels
        path_key: Key in batch containing image path/filename
        base_path: Base directory path for images (optional)
        image_column: Column name for loaded images in output
        num_workers: Number of worker processes for parallel loading
        batch_size: Batch size for parallel image loading
        extension: File extension to append to filenames
        remove_columns: Columns to remove after loading

    Returns:
        datasets.Dataset with 'data' column containing loaded images
    """
    load_fn = partial(
        load_image_batch_lazy,
        path_key=path_key,
        base_path=base_path,
        extension=extension
    )

    cols_to_remove = remove_columns if remove_columns else [path_key]

    lazy_dataset = dataset.map(
        load_fn,
        remove_columns=cols_to_remove,
        num_proc=num_workers,
        batched=True,
        batch_size=batch_size
    )

    return lazy_dataset


def map_dataset_to_images(dataset, image_path, num_workers):
    """Map dataset to load images in batch style.

    Args:
        dataset: datasets.Dataset with 'image_file' and 'labels' columns
        image_path: Root path to image files
        num_workers: Number of worker processes for parallel loading

    Returns:
        datasets.Dataset with 'data' (RGB images) and 'labels' columns
    """
    image_path = Path(image_path)

    def load_image_batch(batch):
        """Load images for a batch of file paths."""
        return {
            'data': [
                Image.open(image_path / f'{image_file}.jpg').convert('RGB')
                for image_file in batch['image_file']
            ]
        }

    return dataset.map(
        load_image_batch,
        remove_columns=['image_file'],
        num_proc=num_workers,
        batched=True,
        batch_size=128
    )


def dataset_split(
    train_images,
    train_labels,
    true_labels,
    split_per: float = 0.9,
    random_seed: int = 256
) -> Tuple:
    """Split dataset into training and validation sets.

    Args:
        train_images: Array of training images
        train_labels: Array of training labels
        true_labels: Array of ground truth labels
        split_per: Proportion of data for training (default: 0.9)
        random_seed: Random seed for reproducibility (default: 1)
        num_classes: Number of classes (unused, kept for API compatibility)

    Returns:
        Tuple of (train_images, train_labels, train_true_labels,
                  val_images, val_labels, val_true_labels)
    """
    np.random.seed(random_seed)
    num_samples = train_labels.shape[0]
    shuffled_indices = np.random.permutation(num_samples)

    n_train = int(num_samples * split_per)
    train_indices = shuffled_indices[:n_train]
    val_indices = shuffled_indices[n_train:]

    return (
        train_images[train_indices],
        train_labels[train_indices],
        true_labels[train_indices],
        train_images[val_indices],
        train_labels[val_indices],
        true_labels[val_indices]
    )


def generate_noisy_labels(
    labels: np.ndarray,
    noise_type: str,
    noise_rate: float,
    nb_classes: int,
    random_seed: int = 256
) -> np.ndarray:
    """Generate noisy labels based on specified noise type.

    Args:
        labels: Original label matrix
        noise_type: Type of noise ('symmetric' or 'pairflip')
        noise_rate: Proportion of labels to corrupt
        nb_classes: Number of classes
        random_seed: Random seed for reproducibility

    Returns:
        Noisy label matrix
    """
    labels = labels.copy()
    labels[labels == 0] = 1
    labels[labels == -1] = 0

    if noise_type == 'symmetric':
        noisy_labels, _, _ = noisify_symmetric(
            labels, noise_rate, random_state=random_seed, nb_classes=nb_classes
        )
    else:
        noisy_labels, _, _ = noisify_pairflip(
            labels, noise_rate, random_state=random_seed, nb_classes=nb_classes
        )
    return noisy_labels


def multiclass_noisify(
    y: np.ndarray,
    P: np.ndarray,
    random_state: Optional[int] = 256
) -> Tuple[np.ndarray, int, int]:
    """Flip classes according to transition probability matrix P.

    Args:
        y: Label matrix (N samples x L labels)
        P: Transition probability matrix
        random_state: Random seed

    Returns:
        Tuple of (noisy_label_matrix, noise_count, total_labels)
    """
    np.random.seed(random_state)

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m, l = y.shape[0], y.shape[1]
    new_y = np.ones((m, l))
    noise_count = 0
    total_label = 0

    for i in range(m):
        label = np.array(y[i], dtype='int')
        idx_label = np.where(label == 1)[0]

        # Iteratively flip labels until stable
        iteration = 0
        max_iterations = 1000
        new_a = None

        while iteration < max_iterations:
            new_a = np.zeros((1, l))
            iteration += 1

            for idx in range(int(idx_label.shape[0])):
                k = idx_label[idx]
                flipped = np.random.multinomial(1, P[k, :], 1)[0]
                flipped = flipped.reshape(1, l)
                new_a += flipped

            new_a = np.array(new_a, dtype='int')
            idx_label_ = np.where(new_a == 1)[0]

            if idx_label_.shape[0] == idx_label.shape[0]:
                break

        if new_a is not None:
            new_y[i, :] = new_a[0, :]
            b = np.sum(new_a.astype('int') != label.astype('int')) / 2
            noise_count += b
            total_label += idx_label.shape[0]

    return new_y, noise_count, total_label


def noisify_symmetric(
    y_train: np.ndarray,
    noise: float,
    random_state: Optional[int] = 256,
    nb_classes: int = 20
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Apply symmetric noise to labels.

    Symmetric noise uniformly flips labels to any other class.

    Args:
        y_train: Original label matrix
        noise: Noise rate (0.0 to 1.0)
        random_state: Random seed
        nb_classes: Number of classes

    Returns:
        Tuple of (noisy_labels, actual_noise_rate, transition_matrix)
    """
    P = np.ones((nb_classes, nb_classes)) * (noise / (nb_classes - 1))

    if noise > 0.0:
        # Set diagonal elements
        np.fill_diagonal(P, 1. - noise)

        y_train_noisy, noise_count, total = multiclass_noisify(
            y_train, P=P, random_state=random_state
        )
        actual_noise = noise_count / total if total > 0 else 0.0
    else:
        y_train_noisy = y_train
        actual_noise = 0.

    return y_train_noisy, actual_noise, P


def noisify_pairflip(
    y_train: np.ndarray,
    noise: float,
    random_state: Optional[int] = 256,
    nb_classes: int = 20
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Apply pairflip noise to labels (adjacent class flipping).

    Pairflip noise flips labels to adjacent classes in a cyclic manner.

    Args:
        y_train: Original label matrix
        noise: Noise rate (0.0 to 1.0)
        random_state: Random seed
        nb_classes: Number of classes

    Returns:
        Tuple of (noisy_labels, actual_noise_rate, transition_matrix)
    """
    P = np.eye(nb_classes)

    if noise > 0.0:
        # Create cyclic pairflip transitions
        for i in range(nb_classes):
            P[i, i] = 1. - noise
            P[i, (i + 1) % nb_classes] = noise

        y_train_noisy, noise_count, total = multiclass_noisify(
            y_train, P=P, random_state=random_state
        )
        actual_noise = noise_count / total if total > 0 else 0.0
    else:
        y_train_noisy = y_train
        actual_noise = 0.

    return y_train_noisy, actual_noise, P


def download_url(url: str, destination: str):
    """Download a file from a URL.

    Args:
        url: URL to download from
        destination: Local path to save the file
    """
    import urllib.request
    print(f'Downloading: "{url}" to {destination}')
    urllib.request.urlretrieve(url, destination)


# Re-export for convenience
__all__ = [
    'load_image',
    'load_image_from_parts',
    'load_image_batch_lazy',
    'create_lazy_dataset',
    'map_dataset_to_images',
    'dataset_split',
    'generate_noisy_labels',
    'multiclass_noisify',
    'noisify_symmetric',
    'noisify_pairflip',
    'download_url',
]
