import csv
import numpy as np
import os
import pandas as pd
import tarfile
from numpy.testing import assert_array_almost_equal
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse

try:
    import datasets
except ImportError:
    datasets = None

try:
    import torch.utils.data as data
except ImportError:
    data = None

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


object_categories = [
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]

VOC2007_URLS = {
    'devkit': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_18-May-2011.tar',
    'trainval': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
    'test_images': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
    'test_anno': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtestnoimgs_06-Nov-2007.tar',
}

VOC2012_URLS = {
    'devkit': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar',
    'trainval': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
    'test_images': 'http://pjreddie.com/media/files/VOC2012test.tar',
    'test_anno': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtestnoimgs_06-Nov-2012.tar',
}


def read_image_label(filepath):
    """Read image labels from a text file.
    
    Args:
        filepath: Path to the label file.
        
    Returns:
        Dictionary mapping image names to labels.
    """
    print(f'[dataset] read {filepath}')
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.split()
            name = parts[0]
            label = int(parts[-1])
            data[name] = label
    return data


def read_object_labels(root, dataset, split):
    """Read object labels for a dataset split.
    
    Args:
        root: Root directory path.
        dataset: Dataset name (e.g., 'VOC2007').
        split: Split name (e.g., 'train', 'test').
        
    Returns:
        Dictionary mapping image names to label arrays.
    """
    labels_path = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
    labeled_data = {}
    num_classes = len(object_categories)

    for i in range(num_classes):
        filepath = os.path.join(labels_path, f'{object_categories[i]}_{split}.txt')
        data = read_image_label(filepath)

        if i == 0:
            for name, label in data.items():
                labels = np.zeros(num_classes)
                labels[i] = label
                labeled_data[name] = labels
        else:
            for name, label in data.items():
                labeled_data[name][i] = label

    return labeled_data


def write_object_labels_csv(filepath, labeled_data):
    """Write object labels to a CSV file using pandas.
    
    Args:
        filepath: Output CSV file path.
        labeled_data: Dictionary mapping image names to label arrays.
    """
    print(f'[dataset] write file {filepath}')
    
    # Convert dictionary to DataFrame
    rows = []
    for name, labels in labeled_data.items():
        row = {'name': name}
        for i, category in enumerate(object_categories):
            row[category] = int(labels[i])
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Write to CSV with proper formatting
    df.to_csv(filepath, index=False, columns=['name'] + object_categories)


def read_object_labels_csv(filepath, header=True):
    """Read object labels from a CSV file.
    
    Args:
        filepath: Path to the CSV file.
        header: Whether the CSV has a header row.
        
    Returns:
        Tuple of (image_names, label_arrays).
    """
    images = []
    labels_list = []
    num_categories = 0
    
    print(f'[dataset] read {filepath}')
    rownum = 0
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        
        for row in reader:
            if header and rownum == 0:
                rownum += 1
                continue
            
            if num_categories == 0:
                num_categories = len(row) - 1
            
            name = row[0]
            labels = np.asarray(row[1:num_categories + 1]).astype(np.float32)
            
            images.append(name)
            labels_list.append(labels)
            rownum += 1
    
    return np.array(images), np.stack(labels_list).astype(np.int32)


def download_voc2007(root):
    """Download VOC2007 dataset.
    
    Args:
        root: Root directory to download to.
    """
    devkit_path = os.path.join(root, 'VOCdevkit')
    images_path = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
    tmpdir = os.path.join(root, 'tmp')

    # Create directories
    os.makedirs(root, exist_ok=True)
    os.makedirs(tmpdir, exist_ok=True)

    # Download and extract devkit
    if not os.path.exists(devkit_path):
        _download_and_extract(VOC2007_URLS['devkit'], tmpdir, root)

    # Download and extract train/val images
    if not os.path.exists(images_path):
        _download_and_extract(VOC2007_URLS['trainval'], tmpdir, root)

    # Download test annotations
    test_anno = os.path.join(devkit_path, 'VOC2007/ImageSets/Main/aeroplane_test.txt')
    if not os.path.exists(test_anno):
        _download_and_extract(VOC2007_URLS['test_anno'], tmpdir, root)

    # Download test images
    test_image = os.path.join(devkit_path, 'VOC2007/JPEGImages/000001.jpg')
    if not os.path.exists(test_image):
        _download_and_extract(VOC2007_URLS['test_images'], tmpdir, root)


def download_voc2012(root):
    """Download VOC2012 dataset.
    
    Args:
        root: Root directory to download to.
    """
    devkit_path = os.path.join(root, 'VOCdevkit')
    images_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'JPEGImages')
    tmpdir = os.path.join(root, 'tmp')

    # Create directories
    os.makedirs(root, exist_ok=True)
    os.makedirs(tmpdir, exist_ok=True)

    # Download and extract devkit
    if not os.path.exists(devkit_path):
        _download_and_extract(VOC2012_URLS['devkit'], tmpdir, root)

    # Download and extract train/val images
    if not os.path.exists(images_path):
        _download_and_extract(VOC2012_URLS['trainval'], tmpdir, root)


def _download_and_extract(url, tmpdir, extract_to):
    """Download and extract a tar file.
    
    Args:
        url: URL to download from.
        tmpdir: Temporary directory for downloads.
        extract_to: Directory to extract files to.
    """
    filename = os.path.basename(urlparse(url).path)
    cached_file = os.path.join(tmpdir, filename)

    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        try:
            import wget
            wget.download(url, cached_file)
        except ImportError:
            import urllib.request
            urllib.request.urlretrieve(url, cached_file)

    # Extract file
    print(f'[dataset] Extracting tar file {cached_file} to {extract_to}')
    cwd = os.getcwd()
    try:
        tar = tarfile.open(cached_file, "r")
        os.chdir(extract_to)
        tar.extractall()
        tar.close()
    finally:
        os.chdir(cwd)
    print('[dataset] Done!')


class Voc2007(data.Dataset):
    """VOC2007 dataset for multi-label classification.
    
    Args:
        root: Root directory path containing VOCdevkit.
        noise_type: Type of noise ('symmetric' or 'pairflip').
        noise_rate: Noise injection rate (0.0 to 1.0).
        split_per: Train/val split ratio.
        nb_classes: Number of classes for noise generation.
        num_workers: Number of workers for data loading.
        noisy_val: Whether to inject noise into validation set.
    """
    
    def __init__(
        self, 
        root,  
        noise_type='symmetric', 
        noise_rate=0.3, 
        split_per=0.9, 
        nb_classes=20, 
        num_workers=4,
        noisy_val=False
    ):
        self.root = root
        self.random_seed = 1
        self.num_workers = num_workers
        self.path_devkit = os.path.join(root, 'VOCdevkit')
        self.path_images = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
        self.classes = object_categories
        self.num_classes = len(self.classes)
        
        # Download dataset
        download_voc2007(self.root)

        # Define CSV file paths
        path_csv = os.path.join(self.root, 'files', 'VOC2007')
        os.makedirs(path_csv, exist_ok=True)
        
        # Generate CSV files if they don't exist
        for phase in ['trainval', 'test']:
            file_csv = os.path.join(path_csv, f'classification_{phase}.csv')
            if not os.path.exists(file_csv):
                labeled_data = read_object_labels(self.root, 'VOC2007', phase)
                write_object_labels_csv(file_csv, labeled_data)

        # Load data from CSV files
        test_images, test_true_labels = self._load_csv_data(
            os.path.join(path_csv, 'classification_test.csv')
        )
        
        trainval_images, trainval_true_labels = self._load_csv_data(
            os.path.join(path_csv, 'classification_trainval.csv')
        )

        # Filter to only existing images
        test_images, test_true_labels = self._filter_existing_images(
            test_images, test_true_labels
        )
        
        trainval_images, trainval_true_labels = self._filter_existing_images(
            trainval_images, trainval_true_labels
        )
        
        print(f'Filtered image size: test: {len(test_images)}, trainval: {len(trainval_images)}')

        # Split trainval into train and validation
        train_images, train_true_labels, val_images, val_true_labels = \
            self._split_dataset(trainval_images, trainval_true_labels, split_per)

        # Generate noisy training labels
        train_labels = self._prepare_labels(
            train_true_labels, noise_type, noise_rate, nb_classes
        )
        
        # Create training dataset
        self.train_data = self._create_dataset(
            train_images, train_labels, 'train', num_workers
        )

        # Prepare validation labels
        val_labels = self._prepare_labels(
            val_true_labels, noise_type, noise_rate, nb_classes, 
            apply_noise=noisy_val
        )

        # Split validation into two halves
        val_images0, val_images1, val_labels0, val_labels1 = train_test_split(
            val_images, val_labels, test_size=0.5, random_state=42
        )

        # Create validation datasets
        self.val_data0 = self._create_dataset(
            val_images0, val_labels0, 'val', num_workers
        )
        
        self.val_data1 = self._create_dataset(
            val_images1, val_labels1, 'val', num_workers
        )

        # Create test dataset
        self.test_data = self._create_dataset(
            test_images, test_true_labels, 'test', num_workers
        )
    
    def _load_csv_data(self, filepath):
        """Load data from CSV file."""
        images, labels = read_object_labels_csv(filepath)
        # Replace 0 with 1 for compatibility
        labels[labels == 0] = 1
        return images, labels
    
    def _filter_existing_images(self, images, labels):
        """Filter to only include existing image files."""
        indices = [
            os.path.exists(os.path.join(self.path_images, img + ".jpg")) 
            for img in images
        ]
        return images[indices], labels[indices]
    
    def _split_dataset(self, images, labels, split_per):
        """Split dataset into train and validation sets."""
        num_samples = len(labels)
        np.random.seed(self.random_seed)
        train_indices = np.random.choice(
            num_samples, int(num_samples * split_per), replace=False
        )
        val_indices = np.delete(np.arange(len(images)), train_indices)
        
        return (
            images[train_indices], 
            labels[train_indices],
            images[val_indices], 
            labels[val_indices]
        )
    
    def _prepare_labels(self, labels, noise_type, noise_rate, nb_classes, apply_noise=True):
        """Prepare labels with optional noise injection."""
        if apply_noise and noise_rate > 0:
            noisy_labels = generate_noisy_labels(
                labels.copy(), noise_type, noise_rate, nb_classes, self.random_seed
            )
            noisy_labels[noisy_labels == -1] = 0
            return noisy_labels
        else:
            labels_copy = labels.copy()
            labels_copy[labels_copy == -1] = 0
            return labels_copy
    
    def _create_dataset(self, images, labels, phase, num_workers):
        """Create HuggingFace Dataset."""
        if datasets is None:
            raise ImportError("datasets library is required but not installed")
        
        return datasets.Dataset.from_dict({
            'image_file': images, 'labels': labels
        }).map(
            lambda batch: {
                'data': [
                    Image.open(
                        os.path.join(self.path_images, img_file + '.jpg')
                    ).convert('RGB')
                    for img_file in batch['image_file']
                ]
            }, 
            remove_columns=['image_file'],
            num_proc=num_workers,
            batched=True
        )

    def get_number_classes(self):
        """Return the number of classes."""
        return self.num_classes


class Voc2012(data.Dataset):
    """VOC2012 dataset for multi-label classification.
    
    Args:
        root: Root directory path containing VOCdevkit.
        noise_type: Type of noise ('symmetric' or 'pairflip').
        noise_rate: Noise injection rate (0.0 to 1.0).
        split_per: Train/val split ratio.
        nb_classes: Number of classes for noise generation.
        num_workers: Number of workers for data loading.
        noisy_val: Whether to inject noise into validation set.
    """
    
    def __init__(
        self, 
        root,  
        noise_type='symmetric', 
        noise_rate=0.3, 
        split_per=0.9, 
        nb_classes=20, 
        num_workers=4,
        noisy_val=False
    ):
        self.root = root
        self.random_seed = 1
        self.num_workers = num_workers
        self.path_devkit = os.path.join(root, 'VOCdevkit')
        self.path_images = os.path.join(root, 'VOCdevkit', 'VOC2012', 'JPEGImages')
        self.classes = object_categories
        self.num_classes = len(self.classes)
        
        # Download dataset
        download_voc2012(self.root)

        # Define CSV file paths
        path_csv = os.path.join(self.root, 'files', 'VOC2012')
        os.makedirs(path_csv, exist_ok=True)
        
        # Generate CSV files if they don't exist
        for phase in ['trainval', 'val']:
            if phase == 'trainval':
                file_csv = os.path.join(path_csv, 'classification_trainval.csv')
            else:
                file_csv = os.path.join(path_csv, 'classification_test.csv')
                
            if not os.path.exists(file_csv):
                labeled_data = read_object_labels(self.root, 'VOC2012', phase)
                write_object_labels_csv(file_csv, labeled_data)

        # Load data from CSV files
        test_images, test_true_labels = self._load_csv_data(
            os.path.join(path_csv, 'classification_test.csv')
        )
        
        trainval_images, trainval_true_labels = self._load_csv_data(
            os.path.join(path_csv, 'classification_trainval.csv')
        )

        # Split trainval into train and validation
        train_images, train_true_labels, val_images, val_true_labels = \
            self._split_dataset(trainval_images, trainval_true_labels, split_per)

        # Generate noisy training labels
        train_labels = self._prepare_labels(
            train_true_labels, noise_type, noise_rate, nb_classes
        )
        
        # Create training dataset
        self.train_data = self._create_dataset(
            train_images, train_labels, 'train', num_workers
        )

        # Prepare validation labels
        val_labels = self._prepare_labels(
            val_true_labels, noise_type, noise_rate, nb_classes, 
            apply_noise=noisy_val
        )

        # Split validation into two halves
        val_images0, val_images1, val_labels0, val_labels1 = train_test_split(
            val_images, val_labels, test_size=0.5, random_state=42
        )

        # Create validation datasets
        self.val_data0 = self._create_dataset(
            val_images0, val_labels0, 'val', num_workers
        )
        
        self.val_data1 = self._create_dataset(
            val_images1, val_labels1, 'val', num_workers
        )

        # Create test dataset
        self.test_data = self._create_dataset(
            test_images, test_true_labels, 'test', num_workers
        )
    
    def _load_csv_data(self, filepath):
        """Load data from CSV file."""
        images, labels = read_object_labels_csv(filepath)
        # Replace 0 with 1 for compatibility
        labels += 1
        return images, labels
    
    def _split_dataset(self, images, labels, split_per):
        """Split dataset into train and validation sets."""
        num_samples = len(labels)
        np.random.seed(self.random_seed)
        train_indices = np.random.choice(
            num_samples, int(num_samples * split_per), replace=False
        )
        val_indices = np.delete(np.arange(len(images)), train_indices)
        
        return (
            images[train_indices], 
            labels[train_indices],
            images[val_indices], 
            labels[val_indices]
        )
    
    def _prepare_labels(self, labels, noise_type, noise_rate, nb_classes, apply_noise=True):
        """Prepare labels with optional noise injection."""
        if apply_noise and noise_rate > 0:
            noisy_labels = generate_noisy_labels(
                labels.copy(), noise_type, noise_rate, nb_classes, self.random_seed
            )
            noisy_labels[noisy_labels == -1] = 0
            return noisy_labels
        else:
            labels_copy = labels.copy()
            labels_copy[labels_copy == -1] = 0
            return labels_copy
    
    def _create_dataset(self, images, labels, phase, num_workers):
        """Create HuggingFace Dataset."""
        if datasets is None:
            raise ImportError("datasets library is required but not installed")
        
        data = datasets.Dataset.from_dict({
            'image_file': images, 'labels': labels
        })

        return data.map(
            lambda batch: {
                'data': [
                    Image.open(
                        os.path.join(self.path_images, img_file + '.jpg')
                    ).convert('RGB')
                    for img_file in batch['image_file']
                ]
            }, 
            remove_columns=['image_file'],
            num_proc=num_workers,
            batched=True
        )

    def get_number_classes(self):
        """Return the number of classes."""
        return self.num_classes


def generate_noisy_labels(labels, noise_type, noise_rate, nb_classes, random_seed):
    """Generate noisy labels based on noise type and rate.
    
    Args:
        labels: Original label matrix.
        noise_type: Type of noise ('symmetric' or 'pairflip').
        noise_rate: Noise injection rate.
        nb_classes: Number of classes.
        random_seed: Random seed for reproducibility.
        
    Returns:
        Noisy label matrix.
    """
    labels_copy = labels.copy()
    labels_copy[labels_copy == 0] = 1
    labels_copy[labels_copy == -1] = 0
    
    if noise_type == 'symmetric':
        noisy_labels, _, _ = noisify_multiclass_symmetric(
            labels_copy, noise_rate, random_seed, nb_classes
        )
    else:
        noisy_labels, _, _ = noisify_pairflip(
            labels_copy, noise_rate, random_seed, nb_classes
        )
    
    return noisy_labels


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

    # Row stochastic matrix
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
        max_iterations = 1000  # Prevent infinite loop
        new_a = None  # Initialize to prevent unbound variable warning
        
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


def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=20):
    """Inject symmetric noise by flipping labels uniformly.
    
    Args:
        y_train: Original label matrix.
        noise: Noise rate (0.0 to 1.0).
        random_state: Random seed for reproducibility.
        nb_classes: Number of classes.
        
    Returns:
        Tuple of (noisy_labels, actual_noise, transition_matrix).
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # Set diagonal elements (probability of keeping original label)
        for i in range(nb_classes):
            P[i, i] = 1. - n

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


def noisify_pairflip(y_train, noise, random_state=None, nb_classes=20):
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
    n = noise

    if n > 0.0:
        # Create cyclic pairflip transitions
        for i in range(nb_classes):
            P[i, i] = 1. - n
            P[i, (i + 1) % nb_classes] = n

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
