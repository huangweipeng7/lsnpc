import json
import os
import pickle
import subprocess
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

try:
    import datasets
except ImportError:
    datasets = None

try:
    import torch.utils.data as data
except ImportError:
    data = None

from numpy.testing import assert_array_almost_equal


COCO_URLS = {
    'train_img': 'http://images.cocodataset.org/zips/train2014.zip',
    'val_img': 'http://images.cocodataset.org/zips/val2014.zip',
    'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
}


def download_coco2014(root, phase='train'):
    """Download and prepare COCO2014 dataset.
    
    Args:
        root: Root directory to download dataset to.
        phase: Dataset phase ('train' or 'val').
    """
    os.makedirs(root, exist_ok=True)
    tmpdir = os.path.join(root, 'tmp')
    os.makedirs(tmpdir, exist_ok=True)
    
    # Download images
    filename = f'{phase}2014.zip'
    cached_file = os.path.join(tmpdir, filename)
    
    if not os.path.exists(cached_file):
        print(f'Downloading: "{COCO_URLS[phase + "_img"]}" to {cached_file}\n')
        import subprocess
        os.chdir(tmpdir)
        subprocess.call(f'wget {COCO_URLS[phase + "_img"]}', shell=True)
        os.chdir('..')
    
    # Extract images
    img_data = os.path.join(root, f'{phase}2014')
    if not os.path.exists(img_data):
        print(f'[dataset] Extracting zip file {cached_file} to {root}')
        os.system(f'unzip -q {cached_file} -d {root}')
        os.remove(cached_file)
    print('[dataset] Done!')
    
    # Download annotations
    annotations_zip = os.path.join(tmpdir, 'annotations_trainval2014.zip')
    if not os.path.exists(annotations_zip):
        print(f'Downloading: "{COCO_URLS["annotations"]}" to {annotations_zip}\n')
        os.chdir(tmpdir)
        subprocess.call(f'wget {COCO_URLS["annotations"]}', shell=True)
        os.chdir('..')
    
    annotations_data = os.path.join(root, 'annotations')
    if not os.path.exists(annotations_data):
        print(f'[dataset] Extracting zip file {annotations_zip} to {root}')
        os.system(f'unzip -q {annotations_zip} -d {root}')
    print('[annotation] Done!')
    
    # Process annotations
    anno_file = os.path.join(root, f'{phase}_anno.json')
    if not os.path.exists(anno_file):
        _process_annotations(root, phase)
    print('[json] Done!')


def _process_annotations(root, phase):
    """Process COCO annotations into simplified JSON format.
    
    Args:
        root: Root directory containing annotations.
        phase: Dataset phase ('train' or 'val').
    """
    annotations_file = os.path.join(root, 'annotations', f'instances_{phase}2014.json')
    
    with open(annotations_file) as f:
        annotations_data = json.load(f)
    
    # Build category mapping
    category_id = {cat['id']: cat['name'] for cat in annotations_data['categories']}
    cat2idx = {cat: idx for idx, cat in enumerate(sorted(category_id.values()))}
    
    # Build image to labels mapping
    annotations_id = {}
    for annotation in annotations_data['annotations']:
        img_id = annotation['image_id']
        if img_id not in annotations_id:
            annotations_id[img_id] = set()
        cat_name = category_id[annotation['category_id']]
        annotations_id[img_id].add(cat2idx[cat_name])
    
    # Build final image list
    img_id_map = {}
    for img in annotations_data['images']:
        if img['id'] not in annotations_id:
            continue
        img_id_map[img['id']] = {
            'file_name': img['file_name'],
            'labels': list(annotations_id[img['id']])
        }
    
    # Save processed annotations
    anno_list = list(img_id_map.values())
    anno_file = os.path.join(root, f'{phase}_anno.json')
    with open(anno_file, 'w') as f:
        json.dump(anno_list, f)
    
    # Save labels-only format
    labels_list = [item['labels'] for item in anno_list]
    labels_file = os.path.join(root, f'{phase}_anno2.json')
    with open(labels_file, 'w') as f:
        json.dump(labels_list, f)
    
    # Save category mapping
    category_file = os.path.join(root, 'category.json')
    if not os.path.exists(category_file):
        with open(category_file, 'w') as f:
            json.dump(cat2idx, f)


class COCO2014(data.Dataset):
    """COCO2014 dataset for multi-label classification.
    
    Expected directory structure after download:
        [root]/
            ├── train2014/
            ├── val2014/
            ├── annotations/
            ├── train_anno.json
            ├── train_anno2.json
            ├── val_anno.json
            ├── val_anno2.json
            └── category.json
    
    Args:
        root: Root directory path containing COCO dataset.
        phase: Dataset phase (unused, kept for compatibility).
        noise_type: Type of noise ('symmetric' or 'pairflip').
        noise_rate: Noise injection rate (0.0 to 1.0).
        split_per: Train/validation split ratio (unused).
        nb_classes: Number of classes (default: 80).
        num_workers: Number of workers for data loading.
        noisy_val: Whether to inject noise into validation set.
    """
    
    def __init__(
        self, 
        root,  
        phase='train', 
        noise_type='symmetric', 
        noise_rate=0.3, 
        split_per=0.9, 
        nb_classes=80, 
        num_workers=4,
        noisy_val=False
    ):
        self.root = root
        self.random_seed = 1
        self.num_workers = num_workers
        
        # Download and prepare dataset
        for split in ['train', 'val']:
            download_coco2014(root, split)
        
        # Load category mapping
        category_file = os.path.join(root, 'category.json')
        with open(category_file) as f:
            self.cat2idx = json.load(f)
        
        self.num_classes = len(self.cat2idx)
        
        # Load annotations
        df_dict = self._load_annotations()
        
        # Process training data
        train_data = df_dict['train']
        if noise_rate > 0:
            train_data['labels'] = self._generate_noisy_labels(
                train_data['labels'], noise_type, noise_rate, nb_classes
            )
        
        print(f'Sample training data: {train_data["image_path"][:5]}')
        print(f'Sample training labels: {train_data["labels"][:5]}')
        
        self.train_data = self._create_dataset(
            train_data, 'train', num_workers
        )
        
        # Split validation data into val and test
        val_data = df_dict['val']
        VAL_SPLIT_SIZE = 10_000
        val_size = len(val_data['labels'])
        
        # Load or create validation indices
        indice_file = './data_process/coco_val_indices.pkl'
        if os.path.exists(indice_file):
            print(f'Using existing validation indices')
            with open(indice_file, 'rb') as f:
                val_indices = pickle.load(f)
        else:
            print(f'Creating validation indices pickle file')
            import random
            val_indices = random.sample(range(val_size), VAL_SPLIT_SIZE)
            with open(indice_file, 'wb') as f:
                pickle.dump(val_indices, f)
        
        # Split into validation and test sets
        all_indices = set(range(val_size))
        test_indices = list(all_indices - set(val_indices))
        
        val_data_split = {
            'image_path': [val_data['image_path'][i] for i in val_indices],
            'labels': [val_data['labels'][i] for i in val_indices]
        }
        
        test_data = {
            'image_path': [val_data['image_path'][i] for i in test_indices],
            'labels': [val_data['labels'][i] for i in test_indices]
        }
        
        print(f'Validation size: {len(val_data_split["image_path"])}, Test size: {len(test_data["image_path"])}')
        
        # Process validation data
        if noisy_val and noise_rate > 0:
            print('Noisify val set')
            val_data_split['labels'] = self._generate_noisy_labels(
                val_data_split['labels'], noise_type, noise_rate, nb_classes
            )
        
        val_data_split['labels'] = np.array(val_data_split['labels'])
        val_data_split['labels'][val_data_split['labels'] == -1] = 0
        val_data_split['labels'] = val_data_split['labels'].tolist()
        
        print('Check validation labels:', val_data_split['labels'][:2])
        
        self.val_data = self._create_dataset(
            val_data_split, 'val', num_workers
        )
        
        # Process test data
        test_data['labels'] = np.array(test_data['labels'])
        test_data['labels'][test_data['labels'] == -1] = 0
        test_data['labels'] = test_data['labels'].tolist()
        
        self.test_data = self._create_dataset(
            test_data, 'val', num_workers  # Note: test uses val2014 folder
        )
    
    def _load_annotations(self):
        """Load annotations for train and validation splits."""
        df_dict = {}
        
        for phase in ['train', 'val']:
            anno_file = os.path.join(self.root, f'{phase}_anno.json')
            with open(anno_file) as f:
                img_list = json.load(f)
            
            true_labels = self._get_true_labels(phase)
            
            df_dict[phase] = {
                'image_path': img_list,
                'labels': true_labels
            }
        
        return df_dict
    
    def _get_true_labels(self, phase):
        """Load true labels from preprocessed JSON file."""
        labels_file = os.path.join(self.root, f'{phase}_anno2.json')
        with open(labels_file) as f:
            labels = json.load(f)
        
        true_labels = np.full((len(labels), len(self.cat2idx)), -1, dtype=np.int32)
        for i, label in enumerate(labels):
            true_labels[i, label] = 1
        
        return true_labels
    
    def _create_dataset(self, data_dict, phase, num_workers):
        """Create HuggingFace Dataset from dictionary."""
        if datasets is None:
            raise ImportError("datasets library is required but not installed")
        
        return datasets.Dataset.from_dict(data_dict).map(
            lambda batch: {
                'data': [
                    Image.open(
                        os.path.join(self.root, f'{phase}2014', img_info['file_name'])
                    ).convert('RGB')
                    for img_info in batch['image_path']
                ]
            },
            remove_columns=['image_path'],
            num_proc=num_workers,
            batched=True
        )
    
    def _generate_noisy_labels(self, labels, noise_type, noise_rate, nb_classes):
        """Generate noisy labels based on noise type and rate."""
        labels_copy = labels.copy()
        labels_copy[labels_copy == 0] = 1
        labels_copy[labels_copy == -1] = 0
        
        if noise_type == 'symmetric':
            noisy_labels, _, _ = noisify_multiclass_symmetric(
                labels_copy, noise_rate, self.random_seed, nb_classes
            )
        else:
            noisy_labels, _, _ = noisify_pairflip(
                labels_copy, noise_rate, self.random_seed, nb_classes
            )
        
        return noisy_labels
    
    def get_number_classes(self):
        """Return the number of classes."""
        return self.num_classes


def multiclass_noisify(y, P, random_state=None):
    """Flip classes according to transition probability matrix P.
    
    Args:
        y: Label matrix of shape (n_samples, n_classes).
        P: Transition probability matrix of shape (n_classes, n_classes).
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (noisy_labels, noise_count, total_labels).
    """
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


def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=80):
    """Inject symmetric noise by flipping labels uniformly.
    
    Args:
        y_train: Original label matrix.
        noise: Noise rate (0.0 to 1.0).
        random_state: Random seed for reproducibility.
        nb_classes: Number of classes.
        
    Returns:
        Tuple of (noisy_labels, actual_noise, transition_matrix).
    """
    P = np.ones((nb_classes, nb_classes)) * (noise / (nb_classes - 1))
    
    if noise > 0.0:
        # Set diagonal elements
        np.fill_diagonal(P, 1. - noise)
        
        y_train_noisy, noise_count, total = multiclass_noisify(
            y_train, P=P, random_state=random_state
        )
        actual_noise = noise_count / total if total > 0 else 0.0
        print(f'Actual noise: {actual_noise:.2f}')
        print(f'Transition matrix:\n{P}')
    else:
        y_train_noisy = y_train
        actual_noise = 0.
    
    return y_train_noisy, actual_noise, P


def noisify_pairflip(y_train, noise, random_state=None, nb_classes=80):
    """Inject pairflip noise by flipping to adjacent classes.
    
    Args:
        y_train: Original label matrix.
        noise: Noise rate (0.0 to 1.0).
        random_state: Random seed for reproducibility.
        nb_classes: Number of classes.
        
    Returns:
        Tuple of (noisy_labels, actual_noise, transition_matrix).
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
        print(f'Actual noise: {actual_noise:.2f}')
        print(f'Transition matrix:\n{P}')
    else:
        y_train_noisy = y_train
        actual_noise = 0.
    
    return y_train_noisy, actual_noise, P
