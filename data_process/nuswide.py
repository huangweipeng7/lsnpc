import numpy as np
import os
import pandas as pd
from numpy.testing import assert_array_almost_equal
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



class NUSWide(data.Dataset):
    """NUS-WIDE dataset for multi-label classification.
    
    Download and prepare the dataset from:
    https://www.kaggle.com/datasets/twerwweqweq/nuswide
    
    Expected directory structure:
        [root]/nus_wide/
            ├── train.csv
            ├── test.csv
            └── images/
    
    Args:
        root: Root directory path containing nus_wide folder.
        noise_type: Type of noise ('symmetric' or 'pairflip').
        noise_rate: Noise injection rate (0.0 to 1.0).
        split_per: Train/validation split ratio (unused, kept for compatibility).
        nb_classes: Number of classes (default: 81).
        num_workers: Number of workers for data loading.
        noisy_val: Whether to inject noise into validation set.
    """
    
    def __init__(
        self, 
        root,  
        noise_type='symmetric', 
        noise_rate=0.0, 
        split_per=0.9, 
        nb_classes=81, 
        num_workers=4,
        noisy_val=False
    ):
        self.root = root
        self.random_seed = 1
        self.num_workers = num_workers
        self.num_classes = nb_classes
        
        # Load annotations
        df_dict = self._load_annotations()
        
        # Process training data
        train_data = df_dict['train']
        if noise_rate > 0:
            print('Creating noisy labels for training set...')
            train_data['labels'] = self._generate_noisy_labels(
                train_data['labels'], noise_type, noise_rate, nb_classes
            )
        else:
            train_data['labels'] = np.array(train_data['labels']).tolist()
        
        self.train_data = self._create_dataset(
            train_data, num_workers
        )
        
        # Process validation data
        val_data0 = df_dict['val0']
        val_data1 = df_dict['val1']
        
        if noisy_val and noise_rate > 0:
            print('Noisify val set')
            val_data0['labels'] = self._generate_noisy_labels(
                val_data0['labels'], noise_type, noise_rate, nb_classes
            )
            val_data1['labels'] = self._generate_noisy_labels(
                val_data1['labels'], noise_type, noise_rate, nb_classes
            )
        
        val_data0['labels'] = np.array(val_data0['labels']).tolist()
        val_data1['labels'] = np.array(val_data1['labels']).tolist()
        
        self.val_data0 = self._create_dataset(val_data0, num_workers)
        self.val_data1 = self._create_dataset(val_data1, num_workers)
        
        # Process test data
        test_data = df_dict['test']
        test_data['labels'] = np.array(test_data['labels']).tolist()
        self.test_data = self._create_dataset(test_data, num_workers)
    
    def _load_annotations(self):
        """Load and parse train/val/test splits from CSV files."""
        anno_path = os.path.join(self.root, 'nus_wide')
        
        train_df = pd.read_csv(os.path.join(anno_path, 'train.csv'))
        test_df = pd.read_csv(os.path.join(anno_path, 'test.csv'))
        
        # Get label columns (exclude metadata columns)
        cols = train_df.columns.tolist()
        for col in ['imageid', 'phase', 'num_label']:
            cols.remove(col)
        
        assert len(cols) == self.num_classes
        
        # Prepare train/validation split
        train_paths = [
            os.path.join('images', x + '.jpg') 
            for x in train_df['imageid'].tolist()
        ]
        train_labels = train_df[cols].values
        
        assert train_labels.max() == 1
        assert train_labels.min() == 0
        
        # Split train into train and validation
        train_paths, val_paths, train_labs, val_labs = train_test_split(
            train_paths, train_labels,
            test_size=0.2,
            random_state=42
        )
        
        # Further split validation into val0 and val1
        val_paths0, val_paths1, val_labs0, val_labs1 = train_test_split(
            val_paths, val_labs,
            test_size=0.5,
            random_state=42
        )
        
        return {
            'train': {'image_path': train_paths, 'labels': train_labs},
            'val0': {'image_path': val_paths0, 'labels': val_labs0},
            'val1': {'image_path': val_paths1, 'labels': val_labs1},
            'test': {
                'image_path': [
                    os.path.join('images', x + '.jpg') 
                    for x in test_df['imageid'].tolist()
                ],
                'labels': test_df[cols].values
            }
        }
    
    def _create_dataset(self, data_dict, num_workers):
        """Create HuggingFace Dataset from dictionary."""
        if datasets is None:
            raise ImportError("datasets library is required but not installed")
        
        return datasets.Dataset.from_dict(data_dict).map(
            lambda batch: {
                'data': [
                    Image.open(
                        os.path.join(self.root, 'nus_wide', img_path)
                    ).convert('RGB')
                    for img_path in batch['image_path']
                ]
            },
            remove_columns=['image_path'],
            num_proc=num_workers,
            batched=True
        )
    
    def _generate_noisy_labels(self, labels, noise_type, noise_rate, nb_classes):
        """Generate noisy labels based on noise type and rate."""
        if noise_type == 'symmetric':
            noisy_labels, _, _ = noisify_multiclass_symmetric(
                labels, noise_rate, self.random_seed, nb_classes
            )
        else:
            noisy_labels, _, _ = noisify_pairflip(
                labels, noise_rate, self.random_seed, nb_classes
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


def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=81):
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


def noisify_pairflip(y_train, noise, random_state=None, nb_classes=81):
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


if __name__ == '__main__':
    nuswide = NUSWide(root='data')

