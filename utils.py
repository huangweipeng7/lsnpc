import csv
import math
import numpy as np
import os
import random
import tarfile
import torch
import torch.nn.functional as F
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from urllib.parse import urlparse
from urllib.request import urlretrieve
from typing import Dict, Union
from pprint import pprint
from torch import Tensor
from typing import Tuple


class Warp(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = int(size)
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)

    def __str__(self):
        return self.__class__.__name__ + ' (size={size}, interpolation={interpolation})'.format(size=self.size,
                          
                                                                                                interpolation=self.interpolation)
class MultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, 875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img):
        im_size = img.size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
        ret_img_group = crop_img_group.resize((self.input_size[0], self.input_size[1]), self.interpolation)
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter
        return ret

    def __str__(self):
        return self.__class__.__name__

def download_url(url, destination=None, progress_bar=True):
    """Download a URL to a local file.
    Parameters
    ----------
    url : str
        The URL to download.
    destination : str, None
        The destination of the file. If None is given the file is saved to a temporary directory.
    progress_bar : bool
        Whether to show a command-line progress bar while downloading.
    Returns
    -------
    filename : str
        The location of the downloaded file.
    Notes
    -----
    Progress bar use/example adapted from tqdm documentation: https://github.com/tqdm/tqdm
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
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            filename, _ = urlretrieve(url, filename=destination, reporthook=my_hook(t))
    else:
        filename, _ = urlretrieve(url, filename=destination)


def read_image_label(file):
    print('[dataset] read ' + file)
    data = dict()
    with open(file, 'r') as f:
        for line in f:
            tmp = line.split(' ')
            name = tmp[0]
            label = int(tmp[-1])
            data[name] = label
    return data


def read_object_labels(root, dataset, phase):
    path_labels = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
    labeled_data = dict()
    num_classes = len(object_categories)

    for i in range(num_classes):
        file = os.path.join(path_labels, object_categories[i] + '_' + phase + '.txt')
        data = read_image_label(file)

        if i == 0:
            for (name, label) in data.items():
                labels = np.zeros(num_classes)
                labels[i] = label
                labeled_data[name] = labels
        else:
            for (name, label) in data.items():
                labeled_data[name][i] = label

    return labeled_data


def write_object_labels_csv(file, labeled_data):
    # write a csv file
    print('[dataset] write file %s' % file)
    with open(file, 'w') as csvfile:
        fieldnames = ['name']
        fieldnames.extend(object_categories)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for (name, labels) in labeled_data.items():
            example = {'name': name}
            for i in range(20):
                example[fieldnames[i + 1]] = int(labels[i])
            writer.writerow(example)


def read_object_labels_csv(file, header=True):
    images = []
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                labels = torch.from_numpy((np.asarray(row[1:num_categories + 1])).astype(np.float32))
                item = (name, labels)
                images.append(item)
            rownum += 1
    return images


def label_dependency_capture(label_dependency, labels):
    posterior_pro_y_y = 0.
    for j in range(labels.size(0)):
        for k in range(labels.size(0)):
            if int(labels[k]) != int(labels[j]):
                t = label_dependency[int(labels[j]), int(labels[k])]
                posterior_pro_y_y += t
    return posterior_pro_y_y


def store_results(
    result_dict: Dict[str, Union[int, float]],
    store_file: str = './runs/results.csv'
):
    """ 
    Store results into the root of result folder.
    Append them if the file is already there.

    Args:
        result_dict: dict containing evaluation metrics and results
        store_file: file to store the results
    """

    if 'pretrained_clf' in result_dict:
        # Update post-name, pre-name, post-uid, pre-uid
        result_dict['post_uid'] = result_dict.pop('uid')
        result_dict['pre_uid'] = result_dict.pop('pretrained_clf').split('/')[-1].replace('.pth','')
    else:
        # Update pre-name, pre-uid
        result_dict['post_model'] = ''
        result_dict['post_uid'] = ''
        result_dict['pre_uid'] = result_dict.pop('uid')
    result_dict['pre_model'] = result_dict.pop('clf_name')
    result_dict['img_encoder'] = result_dict.pop('img_encoder')

    cols = [
        'dataset', 'noise_type', 'noise_rate', 'run_index', 'img_encoder',
        'data_split', 'epoch', 'pre_model', 'pre_uid', 'post_model', 'post_uid', 
        'macro_f1', 'micro_f1', 'macro_mAP', 'micro_mAP',
        'lr', 'batch_size', # For hyper param tuning
    ]
    filtered_dict = {k: result_dict[k] for k in cols if k in result_dict}

    df = pd.DataFrame(filtered_dict, index=[0])
    file_exist= os.path.exists(store_file)
    df.to_csv(store_file, mode='a', index=False, header=(not file_exist))

class ConstraintUtils:
    """
    Utility class for enforcing probability constraints on confusion matrices.
    
    Implements the constraint set Z = {Z ∈ R^(K×2K) | Z * 1_(2K) = 1_K, Z ≥ 0}
    where Z = [A | B] is the horizontal concatenation of confusion matrices A and B.
    
    This ensures that the MCM model maintains valid probabilistic interpretations
    throughout training by projecting confusion matrices onto the constraint set.
    """
    
    @staticmethod
    def project_confusion_matrices(A: Tensor, B: Tensor, epsilon: float = 1e-8) -> Tuple[Tensor, Tensor]:
        """
        Project confusion matrices A and B onto the probability constraint set.
        
        The constraint set requires:
        1. All entries of A and B are non-negative (A ≥ 0, B ≥ 0)
        2. Each row of the concatenated matrix [A | B] sums to 1
        
        Args:
            A: Confusion matrix A of shape (K, K) representing positive class influence
            B: Confusion matrix B of shape (K, K) representing negative class influence  
            epsilon: Small constant for numerical stability (default: 1e-8)
            
        Returns:
            Tuple containing:
            - A_projected: Projected confusion matrix A satisfying constraints
            - B_projected: Projected confusion matrix B satisfying constraints
            
        Raises:
            ValueError: If input tensors have incorrect shapes or types
        """
        # Validate input tensors
        ConstraintUtils._validate_input_matrices(A, B)
        
        K = A.shape[0]
        
        # Step 1: Enforce non-negativity constraint
        # All entries must be ≥ 0
        A_projected = torch.clamp(A, min=0.0)
        B_projected = torch.clamp(B, min=0.0)
        
        # Step 2: Concatenate matrices horizontally: Z = [A | B]
        Z = torch.cat([A_projected, B_projected], dim=1)  # Shape: (K, 2K)
        
        # Step 3: Compute row sums
        row_sums = Z.sum(dim=1, keepdim=True)  # Shape: (K, 1)
        
        # Step 4: Handle zero rows (rows that sum to zero after non-negativity)
        # If a row sums to zero, distribute probability uniformly
        zero_mask = row_sums < epsilon
        uniform_value = 1.0 / (2 * K)  # Uniform distribution over 2K entries
        
        # Replace zero rows with uniform distribution
        if torch.any(zero_mask):
            logger.debug(f"Found {zero_mask.sum().item()} zero rows, applying uniform distribution")
            Z_modified = Z.clone()
            Z_modified[zero_mask.squeeze()] = uniform_value
            row_sums_modified = Z_modified.sum(dim=1, keepdim=True)
        else:
            Z_modified = Z
            row_sums_modified = row_sums
        
        # Step 5: Normalize rows to sum to 1
        # Use safe division with epsilon to prevent division by zero
        Z_normalized = Z_modified / (row_sums_modified + epsilon)
        
        # Step 6: Split back into A and B matrices
        A_projected = Z_normalized[:, :K]  # First K columns
        B_projected = Z_normalized[:, K:]  # Last K columns
        
        # Validate output satisfies constraints
        ConstraintUtils._validate_constraints(A_projected, B_projected, epsilon)
        
        logger.debug(f"Projected confusion matrices: "
                    f"A_range=[{A_projected.min().item():.3f}, {A_projected.max().item():.3f}], "
                    f"B_range=[{B_projected.min().item():.3f}, {B_projected.max().item():.3f}]")
        
        return A_projected, B_projected
    
    @staticmethod
    def _validate_input_matrices(A: Tensor, B: Tensor) -> None:
        """
        Validate input confusion matrices for correct shape and type.
        
        Args:
            A: Confusion matrix A
            B: Confusion matrix B
            
        Raises:
            ValueError: If matrices have incorrect shapes, types, or device mismatch
        """
        if not isinstance(A, Tensor):
            raise ValueError(f"A must be a torch.Tensor, got {type(A)}")
        if not isinstance(B, Tensor):
            raise ValueError(f"B must be a torch.Tensor, got {type(B)}")
        
        if A.dim() != 2:
            raise ValueError(f"A must be 2-dimensional, got shape {A.shape}")
        if B.dim() != 2:
            raise ValueError(f"B must be 2-dimensional, got shape {B.shape}")
        
        if A.shape != B.shape:
            raise ValueError(f"A and B must have same shape, got A{A.shape} vs B{B.shape}")
        
        K = A.shape[0]
        if A.shape[1] != K:
            raise ValueError(f"A must be square matrix (K, K), got shape {A.shape}")
        
        # Check device compatibility
        if A.device != B.device:
            raise ValueError(f"A and B must be on same device, got A{A.device} vs B{B.device}")
        
        logger.debug(f"Validated input matrices: shape={A.shape}, device={A.device}")
    
    @staticmethod
    def _validate_constraints(A: Tensor, B: Tensor, epsilon: float = 1e-6) -> None:
        """
        Validate that projected matrices satisfy all constraints.
        
        Args:
            A: Projected confusion matrix A
            B: Projected confusion matrix B
            epsilon: Tolerance for constraint validation
            
        Raises:
            RuntimeError: If constraints are not satisfied within tolerance
        """
        K = A.shape[0]
        
        # Check non-negativity constraint
        if torch.any(A < -epsilon) or torch.any(B < -epsilon):
            min_A = A.min().item()
            min_B = B.min().item()
            raise RuntimeError(f"Non-negativity constraint violated: A_min={min_A:.6f}, B_min={min_B:.6f}")
        
        # Check row-sum constraint
        Z = torch.cat([A, B], dim=1)
        row_sums = Z.sum(dim=1)
        target_sums = torch.ones(K, device=A.device)
        
        max_deviation = torch.abs(row_sums - target_sums).max().item()
        if max_deviation > epsilon:
            raise RuntimeError(f"Row-sum constraint violated: max_deviation={max_deviation:.6f} > epsilon={epsilon}")
        
        logger.debug(f"Constraint validation passed: max_deviation={max_deviation:.6f}")
    
    @staticmethod
    def compute_constraint_violation(A: Tensor, B: Tensor) -> dict:
        """
        Compute the degree of constraint violation for given confusion matrices.
        
        Args:
            A: Confusion matrix A
            B: Confusion matrix B
            
        Returns:
            Dictionary containing constraint violation metrics:
            - 'non_negativity_violation_A': Maximum negative value in A
            - 'non_negativity_violation_B': Maximum negative value in B  
            - 'row_sum_max_deviation': Maximum deviation from row-sum constraint
            - 'row_sum_mean_deviation': Mean deviation from row-sum constraint
        """
        ConstraintUtils._validate_input_matrices(A, B)
        
        K = A.shape[0]
        
        # Non-negativity violations
        neg_A = torch.clamp(-A, min=0.0)  # Positive where A < 0
        neg_B = torch.clamp(-B, min=0.0)  # Positive where B < 0
        
        non_neg_violation_A = neg_A.max().item()
        non_neg_violation_B = neg_B.max().item()
        
        # Row-sum violations
        Z = torch.cat([A, B], dim=1)
        row_sums = Z.sum(dim=1)
        target_sums = torch.ones(K, device=A.device)
        
        deviations = torch.abs(row_sums - target_sums)
        row_sum_max_deviation = deviations.max().item()
        row_sum_mean_deviation = deviations.mean().item()
        
        violations = {
            'non_negativity_violation_A': non_neg_violation_A,
            'non_negativity_violation_B': non_neg_violation_B,
            'row_sum_max_deviation': row_sum_max_deviation,
            'row_sum_mean_deviation': row_sum_mean_deviation,
            'is_valid': (non_neg_violation_A <= 1e-6 and 
                        non_neg_violation_B <= 1e-6 and 
                        row_sum_max_deviation <= 1e-6)
        }
        
        return violations
    
    @staticmethod
    def create_valid_initialization(K: int, device: torch.device = None) -> Tuple[Tensor, Tensor]:
        """
        Create valid initial confusion matrices satisfying all constraints.
        
        Creates matrices that satisfy:
        - A initialized as identity matrix (no noise assumption)
        - B initialized as zero matrix
        - All constraints satisfied
        
        Args:
            K: Number of classes
            device: Device for created tensors (default: CPU)
            
        Returns:
            Tuple of (A_initial, B_initial) satisfying constraints
        """
        if device is None:
            device = torch.device('cpu')
        
        # Initialize A as identity matrix (no noise assumption)
        A_initial = torch.eye(K, device=device)
        
        # Initialize B as zero matrix
        B_initial = torch.zeros(K, K, device=device)
        
        # Verify constraints are satisfied
        violations = ConstraintUtils.compute_constraint_violation(A_initial, B_initial)
        if not violations['is_valid']:
            raise RuntimeError(f"Initialization violates constraints: {violations}")
        
        
        return A_initial, B_initial