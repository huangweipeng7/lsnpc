from pathlib import Path

import numpy as np
from numpy.testing import assert_array_almost_equal
from PIL import Image
from sklearn.model_selection import train_test_split
import datasets
import torch.utils.data as data


def category_to_idx(categories):
    """Convert category names to index mapping.

    Args:
        categories: List of category names.

    Returns:
        Dictionary mapping category names to indices.
    """
    return {cat: idx for idx, cat in enumerate(categories)}


class DeepFashion(data.Dataset):
    """DeepFashion dataset for multi-label classification.

    Expected directory structure:
        [root]/deepfashion/
            └── Anno_fine/
                ├── train.txt
                ├── train_attr.txt
                ├── val.txt
                ├── val_attr.txt
                ├── test.txt
                └── test_attr.txt

    Args:
        root: Root directory path containing deepfashion folder.
        noise_type: Type of noise ('symmetric' or 'pairflip').
        noise_rate: Noise injection rate (0.0 to 1.0).
        split_per: Train/validation split ratio (unused, kept for compatibility).
        nb_classes: Number of classes (default: 26).
        num_workers: Number of workers for data loading.
        noisy_val: Whether to inject noise into validation set (val).
                   clean_val is NEVER noisified.
    """

    def __init__(
        self,
        root,
        noise_type='symmetric',
        noise_rate=0.0,
        split_per=0.9,
        nb_classes=26,
        num_workers=4,
        noisy_val=False
    ):
        self.root = root
        self.random_seed = 256
        self.num_workers = num_workers

        # Define fashion attributes
        self.attributes = [
            "floral", "graphic", "striped", "embroidered", "pleated",
            "solid", "lattice", "long_sleeve", "short_sleeve", "sleeveless",
            "maxi_length", "mini_length", "no_dress", "crew_neckline",
            "v_neckline", "square_neckline", "no_neckline", "denim",
            "chiffon", "cotton", "leather", "faux", "knit", "tight",
            "loose", "conventional"
        ]
        self.cat2idx = category_to_idx(self.attributes)
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
            train_data['labels'] = np.array(train_data['labels']).tolist()

        self.train_data = self._create_dataset(train_data, num_workers)

        # Process validation data
        val_data = df_dict['val']
        clean_val_data = df_dict['clean_val']

        # Apply noise only to val (based on noisy_val), clean_val is NEVER noisified
        if noisy_val and noise_rate > 0:
            print('Noisify val set')
            val_data['labels'] = self._generate_noisy_labels(
                val_data['labels'], noise_type, noise_rate, nb_classes
            )
        # clean_val remains clean (never noisified regardless of noisy_val)

        val_data['labels'] = np.array(val_data['labels']).tolist()
        clean_val_data['labels'] = np.array(clean_val_data['labels']).tolist()

        self.val_data = self._create_dataset(val_data, num_workers)
        self.clean_val_data = self._create_dataset(clean_val_data, num_workers)

        # Process test data
        test_data = df_dict['test']
        test_data['labels'] = np.array(test_data['labels'])
        test_data['labels'][test_data['labels'] == -1] = 0
        test_data['labels'] = test_data['labels'].tolist()
        self.test_data = self._create_dataset(test_data, num_workers)

    def _load_annotations(self):
        """Load annotations from text files."""
        root = Path(self.root)
        anno_path = root / 'deepfashion' / 'Anno_fine'
        df_dict = {}

        for phase in ['train', 'val', 'test']:
            # Read image paths
            with open(anno_path / f'{phase}.txt') as f:
                img_list = [line.strip() for line in f.readlines()]

            # Read attributes
            with open(anno_path / f'{phase}_attr.txt') as f:
                labels = [
                    [int(i) for i in line.strip().split()]
                    for line in f.readlines()
                ]

            if phase != 'val':
                df_dict[phase] = {'image_path': img_list, 'labels': labels}
            else:
                # Split validation into val and clean_val
                val_img_list, clean_val_img_list, val_labels, clean_val_labels = train_test_split(
                    img_list, labels, test_size=0.5, random_state=256
                )
                df_dict['val'] = {'image_path': val_img_list, 'labels': val_labels}
                df_dict['clean_val'] = {'image_path': clean_val_img_list, 'labels': clean_val_labels}

        return df_dict

    def _create_dataset(self, data_dict, num_workers):
        """Create HuggingFace Dataset from dictionary."""
        root = Path(self.root)

        return datasets.Dataset.from_dict(data_dict).map(
            lambda batch: {
                'data': [
                    Image.open(root / 'deepfashion' / img_path).convert('RGB')
                    for img_path in batch['image_path']
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
            noisy_labels, _, _ = noisify_symmetric(
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


def noisify_symmetric(y_train, noise, random_state=None, nb_classes=26):
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


def noisify_pairflip(y_train, noise, random_state=None, nb_classes=26):
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
