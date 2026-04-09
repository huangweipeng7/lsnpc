import csv
import logging 
import random 
from dataclasses import dataclass, fields, asdict
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Tuple, Union, Optional
from urllib.parse import urlparse
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class WithIndices(Dataset):
    """Dataset wrapper that adds sample indices to each item.
    
    Useful for tracking which samples are being processed during training.
    
    Args:
        dataset: The original PyTorch Dataset.
        index_key: Key under which the sample index will be stored in the dict.
    """
    
    def __init__(self, dataset, index_key='index'):
        self.dataset = dataset
        self.index_key = index_key

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item_with_idx = dict(item)
        item_with_idx[self.index_key] = idx
        return item_with_idx

    def __len__(self):
        return len(self.dataset)


class Warp:
    """Resize images to a fixed size.
    
    Args:
        size: Target size (integer).
        interpolation: PIL interpolation method (default: BILINEAR).
    """
    
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = int(size)
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)

    def __str__(self):
        return (f'{self.__class__.__name__}(size={self.size}, '
                f'interpolation={self.interpolation})')


class MultiScaleCrop:
    """Multi-scale crop augmentation for images.
    
    Performs random cropping at multiple scales with optional fixed crop positions.
    
    Args:
        input_size: Target output size (int or tuple).
        scales: List of scale factors (default: [1, 0.875, 0.75, 0.66]).
        max_distort: Maximum distortion between width and height scales.
        fix_crop: Whether to use fixed crop positions.
        more_fix_crop: Whether to use additional fixed crop positions.
    """
    
    def __init__(
        self, 
        input_size, 
        scales=None, 
        max_distort=1, 
        fix_crop=True, 
        more_fix_crop=True
    ):
        self.scales = scales if scales is not None else [1, 0.875, 0.75, 0.66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = (
            input_size if not isinstance(input_size, int) 
            else [input_size, input_size]
        )
        self.interpolation = Image.BILINEAR

    def __call__(self, img):
        im_size = img.size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = img.crop(
            (offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)
        )
        ret_img_group = crop_img_group.resize(
            (self.input_size[0], self.input_size[1]), 
            self.interpolation
        )
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]
        
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [
            self.input_size[1] if abs(x - self.input_size[1]) < 3 else x 
            for x in crop_sizes
        ]
        crop_w = [
            self.input_size[0] if abs(x - self.input_size[0]) < 3 else x 
            for x in crop_sizes
        ]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        
        if self.fix_crop:
            w_offset, h_offset = self._sample_fix_offset(
                image_w, image_h, crop_pair[0], crop_pair[1]
            )
        else:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(
            self.more_fix_crop, image_w, image_h, crop_w, crop_h
        )
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = [
            (0, 0),                          # upper left
            (4 * w_step, 0),                 # upper right
            (0, 4 * h_step),                 # lower left
            (4 * w_step, 4 * h_step),        # lower right
            (2 * w_step, 2 * h_step),        # center
        ]

        if more_fix_crop:
            ret.extend([
                (0, 2 * h_step),             # center left
                (4 * w_step, 2 * h_step),    # center right
                (2 * w_step, 4 * h_step),    # lower center
                (2 * w_step, 0 * h_step),    # upper center
                (1 * w_step, 1 * h_step),    # upper left quarter
                (3 * w_step, 1 * h_step),    # upper right quarter
                (1 * w_step, 3 * h_step),    # lower left quarter
                (3 * w_step, 3 * h_step),    # lower right quarter
            ])
        return ret

    def __str__(self):
        return self.__class__.__name__


def download_url(
    url: str, 
    destination: str = None, 
    progress_bar: bool = True
) -> str:
    """Download a URL to a local file.
    
    Args:
        url: The URL to download.
        destination: Local file path. If None, saves to temporary directory.
        progress_bar: Whether to show download progress bar.
        
    Returns:
        Path to the downloaded file.
    """
    def my_hook(t):
        last_b = [0]

        def inner(b=1, bsize=1, tsize=None):
            if tsize is not None:
                t.total = tsize
            if b > 0:
                t.update((b - last_b[0]) * bsize)
            last_b[0] = b

        return inner

    if progress_bar:
        with tqdm(
            unit='B', 
            unit_scale=True, 
            miniters=1, 
            desc=url.split('/')[-1]
        ) as t:
            filename, _ = urlretrieve(
                url, 
                filename=destination, 
                reporthook=my_hook(t)
            )
    else:
        filename, _ = urlretrieve(url, filename=destination)
    
    return filename


def read_image_label(file: str) -> Dict[str, int]:
    """Read image-label pairs from a text file.
    
    Expected format: "image_name label" per line.
    
    Args:
        file: Path to the label file.
        
    Returns:
        Dictionary mapping image names to labels.
    """
    print(f'[dataset] read {file}')
    data = {}
    with open(file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                name = parts[0]
                label = int(parts[-1])
                data[name] = label
    return data


def read_object_labels(root: str, dataset: str, phase: str) -> Dict[str, np.ndarray]:
    """Read object labels from VOC-format label files.

    Args:
        root: Root directory of the dataset.
        dataset: Dataset name (e.g., 'VOC2007').
        phase: Data split ('train', 'val', etc.).

    Returns:
        Dictionary mapping image names to label arrays.
    """
    root = Path(root)
    path_labels = root / 'VOCdevkit' / dataset / 'ImageSets' / 'Main'
    labeled_data = {}
    num_classes = len(object_categories)

    for i in range(num_classes):
        file = path_labels / f'{object_categories[i]}_{phase}.txt'
        data = read_image_label(str(file))

        if i == 0:
            for name, label in data.items():
                labels = np.zeros(num_classes)
                labels[i] = label
                labeled_data[name] = labels
        else:
            for name, label in data.items():
                labeled_data[name][i] = label

    return labeled_data


def write_object_labels_csv(filepath: str, labeled_data: Dict[str, np.ndarray]) -> None:
    """Write object labels to a CSV file.
    
    Args:
        filepath: Output CSV file path.
        labeled_data: Dictionary mapping image names to label arrays.
    """
    print(f'[dataset] Writing labels to {filepath}')
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = ['name'] + list(object_categories)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for name, labels in labeled_data.items():
            row = {'name': name}
            for i, category in enumerate(object_categories):
                row[category] = int(labels[i])
            writer.writerow(row)


def read_object_labels_csv(filepath: str, header: bool = True) -> List[Tuple[str, Tensor]]:
    """Read object labels from a CSV file.
    
    Args:
        filepath: Path to the CSV file.
        header: Whether the CSV has a header row.
        
    Returns:
        List of (image_name, label_tensor) tuples.
    """
    images = []
    num_categories = 0
    
    print(f'[dataset] Reading labels from {filepath}')
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        
        for row in reader:
            if header and rownum == 0:
                rownum += 1
                continue
            
            if num_categories == 0:
                num_categories = len(row) - 1
            
            name = row[0]
            labels = torch.from_numpy(
                np.asarray(row[1:num_categories + 1]).astype(np.float32)
            )
            images.append((name, labels))
            rownum += 1
    
    return images


def label_dependency_capture(label_dependency: Tensor, labels: Tensor) -> float:
    """Calculate label dependency score.
    
    Computes the sum of dependency scores between different active labels.
    
    Args:
        label_dependency: Label dependency matrix.
        labels: Binary label vector.
        
    Returns:
        Dependency score.
    """
    posterior_prob = 0.0
    n_labels = labels.size(0)
    
    for j in range(n_labels):
        for k in range(n_labels):
            if labels[k] != labels[j]:
                idx_j = int(labels[j])
                idx_k = int(labels[k])
                posterior_prob += label_dependency[idx_j, idx_k]
    
    return posterior_prob


@dataclass
class TrainingResult:
    """Dataclass for training results."""
    dataset: str = ''
    noise_type: str = ''
    noise_rate: float = 0.0
    run_index: int = 0
    img_encoder: str = ''
    data_split: str = ''
    epoch: int = 0
    pre_model: str = ''
    pre_uid: str = ''
    post_model: str = ''
    post_uid: str = ''
    macro_f1: float = 0.0
    micro_f1: float = 0.0
    mAP: float = 0.0
    micro_mAP: float = 0.0
    lr: float = 0.0
    batch_size: int = 0


def store_results(
    result: Union[TrainingResult, Dict[str, Union[int, float, str]]],
    store_file: str = './runs/results.csv'
) -> None:
    """Store evaluation results to a CSV file.
    
    Appends results to existing file or creates new one.
    
    Args:
        result: TrainingResult instance or dictionary containing evaluation metrics.
        store_file: Path to the results CSV file.
    """
    # Convert dict to TrainingResult if needed
    if isinstance(result, dict):
        result = TrainingResult(**{k: v for k, v in result.items() if k in [f.name for f in fields(TrainingResult)]})
    
    # Get field names for header
    cols = [f.name for f in fields(result)]
    
    # Write directly to file without pandas overhead
    store_file = Path(store_file)
    store_file.parent.mkdir(parents=True, exist_ok=True)
    file_exist = store_file.exists()
    
    with open(store_file, 'a') as f:
        if not file_exist:
            f.write(','.join(cols) + '\n')
        f.write(','.join(str(getattr(result, col, '')) for col in cols) + '\n')


class ConstraintUtils:
    """Utility class for enforcing probability constraints on confusion matrices.
    
    Implements the constraint set:
        Z = {Z ∈ R^(K×2K) | Z * 1_(2K) = 1_K, Z ≥ 0}
    
    where Z = [A | B] is the horizontal concatenation of confusion matrices A and B.
    This ensures valid probabilistic interpretations throughout training.
    """
    
    @staticmethod
    def project_confusion_matrices(
        A: Tensor, 
        B: Tensor, 
        epsilon: float = 1e-8
    ) -> Tuple[Tensor, Tensor]:
        """Project confusion matrices onto the probability constraint set.
        
        Constraints:
            1. All entries non-negative: A ≥ 0, B ≥ 0
            2. Row sums equal 1: [A|B] * 1 = 1
            
        Args:
            A: Confusion matrix A of shape (K, K) for positive class influence.
            B: Confusion matrix B of shape (K, K) for negative class influence.
            epsilon: Small constant for numerical stability.
            
        Returns:
            Tuple of (A_projected, B_projected) satisfying constraints.
            
        Raises:
            ValueError: If input tensors have incorrect shapes or types.
        """
        ConstraintUtils._validate_input_matrices(A, B)
        
        K = A.shape[0]
        
        # Enforce non-negativity
        A_proj = torch.clamp(A, min=1e-6)
        B_proj = torch.clamp(B, min=1e-6)
        
        # Concatenate horizontally
        Z = torch.cat([A_proj, B_proj], dim=1)
        
        # Handle zero rows
        row_sums = Z.sum(dim=1, keepdim=True)
        zero_mask = row_sums < epsilon
        
        if torch.any(zero_mask):
            logger.debug(
                f"Found {zero_mask.sum().item()} zero rows, applying uniform distribution"
            )
            uniform_value = 1.0 / (2 * K)
            Z[zero_mask.squeeze()] = uniform_value
            row_sums = Z.sum(dim=1, keepdim=True)
        
        # Normalize rows
        Z_normalized = Z / (row_sums + epsilon)
        
        assert not torch.any(torch.isnan(Z_normalized)), "NaN detected in normalized matrix"
        
        # Split back
        A_projected = Z_normalized[:, :K]
        B_projected = Z_normalized[:, K:]
        
        # Validate constraints
        ConstraintUtils._validate_constraints(A_projected, B_projected, epsilon)
        
        logger.debug(
            f"Projected matrices: "
            f"A=[{A_projected.min().item():.3f}, {A_projected.max().item():.3f}], "
            f"B=[{B_projected.min().item():.3f}, {B_projected.max().item():.3f}]"
        )
        
        return A_projected, B_projected
    
    @staticmethod
    def _validate_input_matrices(A: Tensor, B: Tensor) -> None:
        """Validate input confusion matrices.
        
        Args:
            A: Confusion matrix A.
            B: Confusion matrix B.
            
        Raises:
            ValueError: If matrices have incorrect shapes, types, or device mismatch.
        """
        if not isinstance(A, Tensor):
            raise ValueError(f"A must be torch.Tensor, got {type(A)}")
        if not isinstance(B, Tensor):
            raise ValueError(f"B must be torch.Tensor, got {type(B)}")
        
        if A.dim() != 2:
            raise ValueError(f"A must be 2D, got shape {A.shape}")
        if B.dim() != 2:
            raise ValueError(f"B must be 2D, got shape {B.shape}")
        
        if A.shape != B.shape:
            raise ValueError(
                f"A and B must have same shape, got A{A.shape} vs B{B.shape}"
            )
        
        K = A.shape[0]
        if A.shape[1] != K:
            raise ValueError(f"A must be square (K, K), got {A.shape}")
        
        if A.device != B.device:
            raise ValueError(
                f"A and B must be on same device, got A{A.device} vs B{B.device}"
            )
        
        logger.debug(f"Validated matrices: shape={A.shape}, device={A.device}")
    
    @staticmethod
    def _validate_constraints(
        A: Tensor, 
        B: Tensor, 
        epsilon: float = 1e-6
    ) -> None:
        """Validate projected matrices satisfy all constraints.
        
        Args:
            A: Projected confusion matrix A.
            B: Projected confusion matrix B.
            epsilon: Tolerance for validation.
            
        Raises:
            RuntimeError: If constraints are violated.
        """
        K = A.shape[0]
        
        # Non-negativity
        if torch.any(A < -epsilon) or torch.any(B < -epsilon):
            raise RuntimeError(
                f"Non-negativity violated: "
                f"A_min={A.min().item():.6f}, B_min={B.min().item():.6f}"
            )
        
        # Row-sum
        Z = torch.cat([A, B], dim=1)
        row_sums = Z.sum(dim=1)
        max_deviation = torch.abs(row_sums - torch.ones(K, device=A.device)).max().item()
        
        if max_deviation > epsilon:
            raise RuntimeError(
                f"Row-sum violated: max_deviation={max_deviation:.6f} > {epsilon}"
            )
        
        logger.debug(f"Constraints satisfied: max_deviation={max_deviation:.6f}")
    
    @staticmethod
    def compute_constraint_violation(
        A: Tensor, 
        B: Tensor
    ) -> Dict[str, Union[float, bool]]:
        """Compute constraint violation metrics.
        
        Args:
            A: Confusion matrix A.
            B: Confusion matrix B.
            
        Returns:
            Dictionary with violation metrics.
        """
        ConstraintUtils._validate_input_matrices(A, B)
        
        K = A.shape[0]
        
        # Non-negativity violations
        neg_A = torch.clamp(-A, min=1e-6)
        neg_B = torch.clamp(-B, min=1e-6)
        
        # Row-sum violations
        Z = torch.cat([A, B], dim=1)
        row_sums = Z.sum(dim=1)
        deviations = torch.abs(row_sums - torch.ones(K, device=A.device))
        
        return {
            'non_negativity_violation_A': neg_A.max().item(),
            'non_negativity_violation_B': neg_B.max().item(),
            'row_sum_max_deviation': deviations.max().item(),
            'row_sum_mean_deviation': deviations.mean().item(),
            'is_valid': (
                neg_A.max().item() <= 1e-6 and 
                neg_B.max().item() <= 1e-6 and 
                deviations.max().item() <= 1e-6
            )
        }
    
    @staticmethod
    def create_valid_initialization(
        K: int, 
        device: torch.device = None
    ) -> Tuple[Tensor, Tensor]:
        """Create valid initial confusion matrices.
        
        Initializes:
            A = I (identity, no noise assumption)
            B = 0 (zero matrix)
        
        Args:
            K: Number of classes.
            device: Device for tensors (default: CPU).
            
        Returns:
            Tuple of (A_initial, B_initial) satisfying constraints.
        """
        if device is None:
            device = torch.device('cpu')
        
        A_initial = torch.eye(K, device=device)
        B_initial = torch.zeros(K, K, device=device)
        
        violations = ConstraintUtils.compute_constraint_violation(
            A_initial, B_initial
        )
        
        if not violations['is_valid']:
            raise RuntimeError(f"Initialization violates constraints: {violations}")
        
        logger.debug(f"Created valid initialization for K={K}")
        return A_initial, B_initial
