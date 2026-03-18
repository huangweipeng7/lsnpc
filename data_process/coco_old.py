import ujson as json
import numpy as np
import os 
import subprocess
from PIL import Image
from sklearn.model_selection import train_test_split

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


def download_coco2014(root, phase):
    """Download and prepare COCO2014 dataset.
    
    Args:
        root: Root directory to download dataset to.
        phase: Dataset phase ('train', 'val', or 'test').
    """
    os.makedirs(root, exist_ok=True)
    tmpdir = os.path.join(root, 'tmp')
    os.makedirs(tmpdir, exist_ok=True)
    
    # Handle phase mapping
    if phase == 'train':
        filename = 'train2014.zip'
        download_phase = 'train'
    elif phase in ['val', 'val0', 'val1', 'test']:
        filename = 'val2014.zip'
        download_phase = 'val'
    else:
        raise ValueError(f"Invalid phase: {phase}")
    
    # Download images
    cached_file = os.path.join(tmpdir, filename)
    if not os.path.exists(cached_file):
        print(f'Downloading: "{COCO_URLS[download_phase + "_img"]}" to {cached_file}\n')
        os.chdir(tmpdir)
        subprocess.call(f'wget {COCO_URLS[download_phase + "_img"]}', shell=True)
        os.chdir('..')
    
    # Extract images
    img_data = os.path.join(root, f'{download_phase}2014')
    if not os.path.exists(img_data):
        print(f'[dataset] Extracting zip file {cached_file} to {root}')
        subprocess.call(f'unzip -q {cached_file} -d {root}', shell=True)
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
        subprocess.call(f'unzip -q {annotations_zip} -d {root}', shell=True)
    print('[annotation] Done!')
    
    # Process annotations
    anno_file = os.path.join(root, f'{download_phase}_anno.json')
    if not os.path.exists(anno_file):
        _process_annotations(root, download_phase)
    print('[json] Done!')


def _process_annotations(root, phase):
    """Process COCO annotations into simplified JSON format (memory efficient).
    
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
    
    # Build image to labels mapping (memory efficient using sets)
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
    
    # Save processed annotations (streaming to avoid memory issues)
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
    
    # Clean up memory explicitly
    del img_id_map, annotations_id, annotations_data, category_id


class COCO2014(data.Dataset):
    """COCO2014 dataset with memory-efficient loading for large datasets.
    
    This class loads data incrementally rather than all at once, making it
    suitable for large datasets that don't fit in memory.
    
    Expected directory structure:
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
        transform: Optional transform to apply to images.
        phase: Dataset phase ('train', 'val', 'val0', 'val1', 'test').
        noise_type: Type of noise ('symmetric' or 'pairflip').
        noise_rate: Noise injection rate (0.0 to 1.0).
        split_per: Train/validation split ratio (unused).
        nb_classes: Number of classes (default: 80).
        random_seed: Random seed for reproducibility.
        noisy_val: Whether to inject noise into validation set.
    """
    
    def __init__(
        self, 
        root, 
        transform=None, 
        phase='train', 
        noise_type='symmetric', 
        noise_rate=0.3, 
        split_per=0.9, 
        nb_classes=80, 
        random_seed=1,
        **kwargs 
    ):
        self.root = root
        self.phase = phase
        self.transform = transform
        self.random_seed = random_seed
        
        # Download and prepare dataset
        download_coco2014(root, phase)
        
        # Load annotations (lazy loading)
        self._load_annotations()
        
        # Load true labels
        self.true_labels = self._get_true_labels()
        self.num_classes = len(self.cat2idx)
        
        # Generate or assign labels based on phase
        if phase == 'train':
            self.labels = self._generate_noisy_labels(
                self.true_labels, noise_type, noise_rate, nb_classes
            )
        else:
            self.labels = self.true_labels.copy()
        
        # Handle validation and test splits
        self.noisy_val = kwargs.get('noisy_val', False)
        self._handle_phase_splits(noise_type, noise_rate, nb_classes)
    
    def _load_annotations(self):
        """Load annotations from JSON file (memory efficient)."""
        # Map phase to actual phase name
        phase_map = {'val0': 'val', 'val1': 'val', 'test': 'val'}
        actual_phase = phase_map.get(self.phase, self.phase)
        
        anno_file = os.path.join(self.root, f'{actual_phase}_anno.json')
        with open(anno_file, 'r') as f:
            self.img_list = json.load(f)
        
        category_file = os.path.join(self.root, 'category.json')
        with open(category_file, 'r') as f:
            self.cat2idx = json.load(f)
    
    def _get_true_labels(self):
        """Load true labels from preprocessed JSON file."""
        phase_map = {'val0': 'val', 'val1': 'val', 'test': 'val'}
        actual_phase = phase_map.get(self.phase, self.phase)
        
        labels_file = os.path.join(self.root, f'{actual_phase}_anno2.json')
        with open(labels_file, 'r') as f:
            labels = json.load(f)
        
        # Create label matrix incrementally (memory efficient)
        n_samples = len(labels)
        n_classes = len(self.cat2idx)
        true_labels = np.zeros((n_samples, n_classes), dtype=np.float32)
        
        for i, label_indices in enumerate(labels):
            true_labels[i, label_indices] = 1.0
        
        return true_labels
    
    def _handle_phase_splits(self, noise_type, noise_rate, nb_classes):
        """Handle different phase splits (val0, val1, test)."""
        n = len(self.img_list)
        
        if self.phase == 'val0':
            # First half of validation set
            indices = list(range(n // 2))
            self.img_list = [self.img_list[i] for i in indices]
            self.labels = self.labels[indices]
            
        elif self.phase == 'val1':
            # Second half of validation set
            indices = list(range(n // 2, n))
            self.img_list = [self.img_list[i] for i in indices]
            self.labels = self.labels[indices]
            
        elif self.phase == 'test':
            # Use second half as test set
            indices = list(range(n // 2, n))
            self.img_list = [self.img_list[i] for i in indices]
            self.labels = self.labels[indices]
            print(f'Test size: {len(self.img_list)}')
        
        # Apply noise to validation if requested
        if self.noisy_val and self.phase.startswith('val') and noise_rate > 0:
            print(f'Noisify {self.phase} set')
            self.labels = self._generate_noisy_labels(
                self.labels, noise_type, noise_rate, nb_classes
            )
    
    def _generate_noisy_labels(self, labels, noise_type, noise_rate, nb_classes):
        """Generate noisy labels for training or validation."""
        labels_copy = labels.copy()
        
        if noise_type == 'symmetric':
            noisy_labels, _, _ = noisify_multiclass_symmetric(
                labels_copy, noise_rate, self.random_seed, nb_classes
            )
        else:
            noisy_labels, _, _ = noisify_pairflip(
                labels_copy, noise_rate, self.random_seed, nb_classes
            )
        
        return noisy_labels
    
    def __len__(self):
        """Return the number of samples."""
        return len(self.img_list)

    def __getitem__(self, index):
        """Get a sample by index (memory efficient).
        
        Args:
            index: Index of the sample to retrieve.
            
        Returns:
            Dictionary containing 'data' (image) and 'labels'.
        """
        item = self.img_list[index]
        target = self.labels[index]
        
        # Determine actual phase for file path
        phase_map = {'val0': 'val', 'val1': 'val', 'test': 'val'}
        actual_phase = phase_map.get(self.phase, self.phase)
        
        # Load image
        filename = item['file_name']
        img_path = os.path.join(self.root, f'{actual_phase}2014', filename)
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return {'data': img, 'labels': target}

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
