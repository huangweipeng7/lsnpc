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
        self.random_seed = 256
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
        val_data = df_dict['val']
        clean_val_data = df_dict['clean_val']

        # Apply noise only to val (not clean_val)
        if noisy_val and noise_rate > 0:
            print('Noisify val set')
            val_data['labels'] = self._generate_noisy_labels(
                val_data['labels'], noise_type, noise_rate, nb_classes
            )
            # clean_val remains clean (never noisified)

        val_data['labels'] = np.array(val_data['labels']).tolist()
        clean_val_data['labels'] = np.array(clean_val_data['labels']).tolist()

        self.val_data = self._create_dataset(val_data, num_workers)
        self.clean_val_data = self._create_dataset(clean_val_data, num_workers)

        # Process test data
        test_data = df_dict['test']
        test_data['labels'] = np.array(test_data['labels']).tolist()
        self.test_data = self._create_dataset(test_data, num_workers)

    def _load_annotations(self):
        """Load and parse train/val/test splits from CSV files."""
        root = Path(self.root)
        anno_path = root / 'nus_wide'

        train_df = pd.read_csv(anno_path / 'train.csv')
        test_df = pd.read_csv(anno_path / 'test.csv')

        # Get label columns (exclude metadata columns)
        cols = train_df.columns.tolist()
        for col in ['imageid', 'phase', 'num_label']:
            cols.remove(col)

        assert len(cols) == self.num_classes

        # Prepare train/validation split
        train_paths = [
            f'images/{x}.jpg'
            for x in train_df['imageid'].tolist()
        ]
        train_labels = train_df[cols].values

        assert train_labels.max() == 1
        assert train_labels.min() == 0

        # Split train into train and validation
        train_paths, val_paths, train_labs, val_labs = train_test_split(
            train_paths, train_labels,
            test_size=0.2,
            random_state=256
        )

        # Further split validation into val and clean_val
        val_paths_split, clean_val_paths, val_labs_split, clean_val_labs = train_test_split(
            val_paths, val_labs,
            test_size=0.5,
            random_state=256
        )

        return {
            'train': {'image_path': train_paths, 'labels': train_labs},
            'val': {'image_path': val_paths_split, 'labels': val_labs_split},
            'clean_val': {'image_path': clean_val_paths, 'labels': clean_val_labs},
            'test': {
                'image_path': [
                    f'images/{x}.jpg'
                    for x in test_df['imageid'].tolist()
                ],
                'labels': test_df[cols].values
            }
        }

    def _create_dataset(self, data_dict, num_workers):
        """Create HuggingFace Dataset from dictionary."""
        root = Path(self.root)

        def load_batch(batch):
            return {
                'data': [
                    load_image(str(root / 'nus_wide' / img_path))
                    for img_path in batch['image_path']
                ]
            }

        return datasets.Dataset.from_dict(data_dict).map(
            load_batch,
            remove_columns=['image_path'],
            num_proc=num_workers,
            batched=True
        )

    def _generate_noisy_labels(self, labels, noise_type, noise_rate, nb_classes):
        """Generate noisy labels based on noise type and rate."""
        if noise_type == 'symmetric':
            noisy_labels, _, _ = noisify_symmetric(
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


if __name__ == '__main__':
    nuswide = NUSWide(root='data')
