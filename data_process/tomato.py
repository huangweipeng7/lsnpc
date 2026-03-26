from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import datasets
import torch.utils.data as data

from .shared_utils import (
    load_image,
    noisify_symmetric,
    noisify_pairflip,
)


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
        self.random_seed = 256
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

        # Process validation data (split into val and clean_val)
        self.val_data = self._process_val_data(
            df_dict['val'], 'val', noise_type, noise_rate,
            nb_classes, noisy_val, num_workers
        )

        self.clean_val_data = self._process_val_data(
            df_dict['clean_val'], 'val', noise_type, noise_rate,
            nb_classes, noisy_val=False, num_workers=num_workers  # clean_val is never noisified
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
        root = Path(self.root)
        anno_path = root / 'tomato' / 'anno.csv'
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
                # Split validation set into val and clean_val
                val_img_list, clean_val_img_list, val_labels, clean_val_labels = \
                    train_test_split(
                        tmp_img_list, tmp_true_labels,
                        test_size=0.5, random_state=256
                    )
                df_dict['val'] = {
                    'image_path': val_img_list,
                    'labels': val_labels
                }
                df_dict['clean_val'] = {
                    'image_path': clean_val_img_list,
                    'labels': clean_val_labels
                }

        return df_dict

    def _get_true_labels(self, df):
        """Extract true labels from dataframe."""
        true_labels = np.array(df[list(self.cat2idx.keys())].values)
        true_labels = np.nan_to_num(true_labels, nan=-1).astype(np.int32)
        return true_labels

    def _create_dataset(self, data_dict, phase, num_workers):
        """Create HuggingFace Dataset from dictionary."""
        root = Path(self.root)

        def load_batch(batch):
            return {
                'data': [
                    load_image(str(root / 'tomato' / phase / img_path))
                    for img_path in batch['image_path']
                ]
            }

        return datasets.Dataset.from_dict(data_dict).map(
            load_batch,
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
            # Note: clean_val is never noisified (noisy_val=False passed explicitly)

        val_data['labels'] = np.array(val_data['labels'])
        val_data['labels'][val_data['labels'] == -1] = 0
        val_data['labels'] = val_data['labels'].tolist()

        return self._create_dataset(val_data, phase, num_workers)

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

    def __len__(self):
        """Return the length of the dataset.

        Note: This method is deprecated. Use len(dataset.train_data) or
        len(dataset.test_data) instead.
        """
        raise NotImplementedError(
            "Use len(dataset.train_data), len(dataset.val_data), "
            "len(dataset.clean_val_data), or len(dataset.test_data) instead."
        )

    def __getitem__(self, index):
        """Get a sample from the dataset.

        Note: This method is deprecated. Access items directly from
        train_data, val_data, clean_val_data, or test_data.
        """
        raise NotImplementedError(
            "Access items directly from train_data, val_data, "
            "clean_val_data, or test_data."
        )

    def get_number_classes(self):
        """Return the number of classes."""
        return self.num_classes
