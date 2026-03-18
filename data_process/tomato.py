import numpy as np
import os
import pandas as pd
from numpy.testing import assert_array_almost_equal
from PIL import Image
from sklearn.model_selection import train_test_split
import torch.utils.data as data

try:
    import datasets
except ImportError:
    datasets = None


def category_to_idx(categories):
    """Convert category names to index mapping.
    
    Args:
        categories: List of category names.
        
    Returns:
        Dictionary mapping category names to indices.
    """
    return {cat: idx for idx, cat in enumerate(categories)}


class Tomato(data.Dataset):
    """Tomato dataset for multi-label classification.
    
    The dataset should be organized in the following structure:
        [root]/tomato/
            ├── train/
            ├── test/
            ├── val/
            └── anno.csv
    
    Args:
        root: Root directory path containing the tomato folder.
        noise_type: Type of noise to inject ('symmetric' or 'pairflip').
        noise_rate: Rate of noise injection (0.0 to 1.0).
        split_per: Split percentage (unused parameter, kept for compatibility).
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
        nb_classes=8, 
        num_workers=4,
        noisy_val=False
    ):
        self.root = root
        self.random_seed = 1
        self.num_workers = num_workers
        
        # Define class categories
        self.categories = [
            'Early blight', 
            'Healthy', 
            'Late blight',
            'Leaf Miner',
            'Magnesium Deficiency',
            'Nitrogen Deficiency',
            'Pottassium Deficiency',
            'Spotted Wilt Virus'
        ]
        self.cat2idx = category_to_idx(self.categories)
        self.num_classes = len(self.cat2idx)
        
        # Load annotations
        df_dict = self._load_annotations()
        
        # Process training data
        train_data = df_dict['train']
        if noise_rate > 0:
            train_data['labels'] = self._generate_noisy_labels(
                train_data['labels'], noise_type, noise_rate, nb_classes
            )
        else:
            train_data['labels'] = np.array(train_data['labels'])
            train_data['labels'][train_data['labels'] == -1] = 0
            train_data['labels'] = train_data['labels'].tolist()
        
        self.train_data = self._create_dataset(
            train_data, 'train', num_workers
        )
        
        # Process validation data (split into val0 and val1)
        self.val_data0 = self._process_val_data(
            df_dict['val0'], 'val', noise_type, noise_rate, 
            nb_classes, noisy_val, num_workers
        )
        
        self.val_data1 = self._process_val_data(
            df_dict['val1'], 'val', noise_type, noise_rate, 
            nb_classes, noisy_val, num_workers
        )
        
        # Process test data
        test_data = df_dict['test']
        test_data['labels'] = np.array(test_data['labels'])
        test_data['labels'][test_data['labels'] == -1] = 0
        test_data['labels'] = test_data['labels'].tolist()
        self.test_data = self._create_dataset(
            test_data, 'test', num_workers
        )
    
    def _load_annotations(self):
        """Load and parse annotation CSV file."""
        anno_path = os.path.join(self.root, 'tomato', 'anno.csv')
        df = pd.read_csv(anno_path)
        
        df_dict = {}
        for phase in ['train', 'val', 'test']: 
            tmp_df = df[df['dataset'] == phase].copy()
            tmp_img_list = tmp_df['filename'].values
            tmp_true_labels = self._get_true_labels(tmp_df)
            
            if phase != 'val':
                df_dict[phase] = {
                    'image_path': tmp_img_list, 
                    'labels': tmp_true_labels
                }
            else:
                # Split validation set into two halves
                val_img_list0, val_img_list1, val_labels0, val_labels1 = \
                    train_test_split(
                        tmp_img_list, tmp_true_labels, 
                        test_size=0.5, random_state=42
                    )
                df_dict['val0'] = {
                    'image_path': val_img_list0, 
                    'labels': val_labels0
                }
                df_dict['val1'] = {
                    'image_path': val_img_list1, 
                    'labels': val_labels1
                }
        
        return df_dict
    
    def _get_true_labels(self, df):
        """Extract true labels from dataframe."""
        true_labels = np.array(df[self.cat2idx.keys()].values)
        true_labels = np.nan_to_num(true_labels, nan=-1).astype(np.int32)
        return true_labels
    
    def _create_dataset(self, data_dict, phase, num_workers):
        """Create HuggingFace Dataset from dictionary."""
        if datasets is None:
            raise ImportError("datasets library is required but not installed")
        
        return datasets.Dataset.from_dict(data_dict).map(
            lambda batch: {
                'data': [
                    Image.open(
                        os.path.join(self.root, 'tomato', phase, img_path)
                    ).convert('RGB')
                    for img_path in iter(batch['image_path'])
                ]
            }, 
            remove_columns=['image_path'],
            num_proc=num_workers,
            batched=True
        )
    
    def _process_val_data(self, val_data, phase, noise_type, noise_rate, 
                          nb_classes, noisy_val, num_workers):
        """Process validation data with optional noise injection."""
        if noisy_val:
            print('Noisify val set')
            val_data['labels'] = self._generate_noisy_labels(
                val_data['labels'], noise_type, noise_rate, nb_classes
            )
        
        val_data['labels'] = np.array(val_data['labels'])
        val_data['labels'][val_data['labels'] == -1] = 0
        val_data['labels'] = val_data['labels'].tolist()
        
        return self._create_dataset(val_data, phase, num_workers)
    
    def _generate_noisy_labels(self, labels, noise_type, noise_rate, nb_classes):
        """Generate noisy labels based on noise type and rate."""
        labels_copy = labels.copy()
        N, nc = labels_copy.shape
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
       
    def __len__(self):
        """Return the length of the dataset.
        
        Note: This method is deprecated. Use len(dataset.train_data) or
        len(dataset.test_data) instead.
        """
        raise NotImplementedError(
            "Use len(dataset.train_data), len(dataset.val_data0), "
            "len(dataset.val_data1), or len(dataset.test_data) instead."
        )

    def __getitem__(self, index):
        """Get a sample from the dataset.
        
        Note: This method is deprecated. Access items directly from
        train_data, val_data0, val_data1, or test_data.
        """
        raise NotImplementedError(
            "Access items directly from train_data, val_data0, "
            "val_data1, or test_data."
        )

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
        Tuple of (noisy_labels, noise_count, total_label).
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


def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=8):
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


def noisify_pairflip(y_train, noise, random_state=None, nb_classes=8):
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
