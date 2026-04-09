import datasets
import numpy as np
import pickle
import random
import subprocess
import torch.utils.data as data
import ujson as json
from pathlib import Path 
from PIL import Image

from .shared_utils import (
    noisify_symmetric,
    noisify_pairflip,
)


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
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tmpdir = root / 'tmp'
    tmpdir.mkdir(parents=True, exist_ok=True)

    # Download images
    filename = f'{phase}2014.zip'
    cached_file = tmpdir / filename

    if not cached_file.exists():
        print(f'Downloading: "{COCO_URLS[phase + "_img"]}" to {cached_file}\n')
        subprocess.run(
            f'wget {COCO_URLS[phase + "_img"]}',
            shell=True,
            cwd=str(tmpdir)
        )

    # Extract images
    img_data = root / f'{phase}2014'
    if not img_data.exists():
        print(f'[dataset] Extracting zip file {cached_file} to {root}')
        subprocess.run(
            f'unzip -q {cached_file} -d {root}',
            shell=True,
            cwd=str(root)
        )
        cached_file.unlink()
    print('[dataset] Done!')

    # Download annotations
    annotations_zip = tmpdir / 'annotations_trainval2014.zip'
    if not annotations_zip.exists():
        print(f'Downloading: "{COCO_URLS["annotations"]}" to {annotations_zip}\n')
        subprocess.run(
            f'wget {COCO_URLS["annotations"]}',
            shell=True,
            cwd=str(tmpdir)
        )

    annotations_data = root / 'annotations'
    if not annotations_data.exists():
        print(f'[dataset] Extracting zip file {annotations_zip} to {root}')
        subprocess.run(
            f'unzip -q {annotations_zip} -d {root}',
            shell=True,
            cwd=str(root)
        )
    print('[annotation] Done!')

    # Process annotations
    anno_file = root / f'{phase}_anno.json'
    if not anno_file.exists():
        _process_annotations(root, phase)
    print('[json] Done!')


def _process_annotations(root, phase):
    """Process COCO annotations into simplified JSON format.

    Args:
        root: Root directory containing annotations.
        phase: Dataset phase ('train' or 'val').
    """
    root = Path(root)
    annotations_file = root / 'annotations' / f'instances_{phase}2014.json'

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
    anno_file = root / f'{phase}_anno.json'
    with open(anno_file, 'w') as f:
        json.dump(anno_list, f)

    # Save labels-only format
    labels_list = [item['labels'] for item in anno_list]
    labels_file = root / f'{phase}_anno2.json'
    with open(labels_file, 'w') as f:
        json.dump(labels_list, f)

    # Save category mapping
    category_file = root / 'category.json'
    if not category_file.exists():
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
        noisy_val: Whether to inject noise into validation set (val).
                   clean_val is NEVER noisified.
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
        self.random_seed = 256
        self.num_workers = num_workers

        # Download and prepare dataset
        for split in ['train', 'val']:
            download_coco2014(root, split)

        root = Path(root)
        # Load category mapping
        category_file = root / 'category.json'
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

        # Split validation data into val, clean_val, and test
        val_data = df_dict['val']
        VAL_SPLIT_SIZE = 10_000
        val_size = len(val_data['labels'])

        # Load or create validation indices
        indice_file = Path('./data_process/coco_val_indices.pkl')
        if indice_file.exists():
            print(f'Using existing validation indices')
            with open(indice_file, 'rb') as f:
                val_indices = pickle.load(f)
        else:
            print(f'Creating validation indices pickle file')
            val_indices = random.sample(range(val_size), VAL_SPLIT_SIZE)
            with open(indice_file, 'wb') as f:
                pickle.dump(val_indices, f)

        # Split into val, clean_val, and test sets
        all_indices = set(range(val_size))
        val_set = set(val_indices)
        test_indices = list(all_indices - val_set)

        # Further split val_indices into val and clean_val (randomized)
        val_indices_list = list(val_set)
        val_indices_array = np.array(val_indices_list)
        noisy_val_indices, clean_val_indices = train_test_split(
            val_indices_array, test_size=0.5, random_state=self.random_seed
        )

        val_data_split = {
            'image_path': [val_data['image_path'][i] for i in noisy_val_indices],
            'labels': [val_data['labels'][i] for i in noisy_val_indices]
        }

        clean_val_data = {
            'image_path': [val_data['image_path'][i] for i in clean_val_indices],
            'labels': [val_data['labels'][i] for i in clean_val_indices]
        }

        test_data = {
            'image_path': [val_data['image_path'][i] for i in test_indices],
            'labels': [val_data['labels'][i] for i in test_indices]
        }

        print(f'Validation size: {len(val_data_split["image_path"])}, '
              f'Clean val size: {len(clean_val_data["image_path"])}, '
              f'Test size: {len(test_data["image_path"])}')

        # Process val data (with optional noise based on noisy_val)
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

        # clean_val is NEVER noisified
        clean_val_data['labels'] = np.array(clean_val_data['labels'])
        clean_val_data['labels'][clean_val_data['labels'] == -1] = 0
        clean_val_data['labels'] = clean_val_data['labels'].tolist()

        self.clean_val_data = self._create_dataset(
            clean_val_data, 'val', num_workers
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
        root = Path(self.root)

        for phase in ['train', 'val']:
            anno_file = root / f'{phase}_anno.json'
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
        root = Path(self.root)
        labels_file = root / f'{phase}_anno2.json'
        with open(labels_file) as f:
            labels = json.load(f)

        true_labels = np.full((len(labels), len(self.cat2idx)), -1, dtype=np.int32)
        for i, label in enumerate(labels):
            true_labels[i, label] = 1

        return true_labels

    def _create_dataset(self, data_dict, phase, num_workers):
        """Create HuggingFace Dataset from dictionary."""
        root = Path(self.root)
        return datasets.Dataset.from_dict(data_dict).map(
            lambda batch: {
                'data': [
                    Image.open(root / f'{phase}2014' / img_info['file_name']).convert('RGB')
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
