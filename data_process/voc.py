"""VOC Dataset Utilities for Multi-Label Classification with Noisy Labels.

This module provides utilities for loading and processing the PASCAL VOC dataset
(2007 and 2012 versions) with support for synthetic noisy label generation,
lazy image loading, and optimized data processing.
"""

import csv
import datasets
import numpy as np
import subprocess
import torch.utils.data as data
from pathlib import Path 
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse

from .shared_utils import ( 
    dataset_split,
    download_url,
    generate_noisy_labels,
    map_dataset_to_images,
)

# PASCAL VOC object categories (20 classes)
OBJECT_CATEGORIES = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']


def read_image_label(file):
    """Read image labels from a text file.

    Args:
        file: Path to the label file (format: "image_name label")

    Returns:
        Dictionary mapping image names to labels
    """
    print(f'[dataset] read {file}')
    data = {}
    with open(file, 'r') as f:
        for line in f:
            parts = line.split()
            name = parts[0]
            label = int(parts[-1])
            data[name] = label
    return data


def read_object_labels(root, dataset, split):
    """Read object labels for all classes from VOC format files.

    Args:
        root: Root directory of VOC dataset
        dataset: Dataset name (e.g., 'VOC2007', 'VOC2012')
        split: Split name (e.g., 'trainval', 'test')

    Returns:
        Dictionary mapping image names to multi-label vectors
    """
    root = Path(root)
    path_labels = root / 'VOCdevkit' / dataset / 'ImageSets' / 'Main'
    labeled_data = {}
    num_classes = len(OBJECT_CATEGORIES)

    for i, category in enumerate(OBJECT_CATEGORIES):
        file = path_labels / f'{category}_{split}.txt'
        data = read_image_label(str(file))

        if i == 0:
            for name, label in data.items():
                labels = np.zeros(num_classes)
                labels[i] = label
                labeled_data[name] = labels
        else:
            for name, label in data.items():
                labeled_data[name][i] = label

    return labeled_data


def write_object_labels_csv(file, labeled_data):
    """Write labeled data to a CSV file.

    Args:
        file: Output CSV file path
        labeled_data: Dictionary mapping image names to label vectors
    """
    print(f'[dataset] write file {file}')
    with open(file, 'w') as csvfile:
        fieldnames = ['name'] + OBJECT_CATEGORIES
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for name, labels in labeled_data.items():
            example = {'name': name}
            for i, label in enumerate(labels):
                example[fieldnames[i + 1]] = int(label)
            writer.writerow(example)


def read_object_labels_csv(file, header=True):
    """Read labeled data from a CSV file.

    Args:
        file: Input CSV file path
        header: Whether the CSV file has a header row

    Returns:
        Tuple of (image_names array, labels tensor)
    """
    images = []
    labels_list = []
    num_categories = 0
    print(f'[dataset] read {file}')

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
                labels = np.asarray(row[1:num_categories + 1]).astype(np.float32)
                labels = torch.from_numpy(labels)
                images.append(name)
                labels_list.append(labels)
            rownum += 1

    return np.stack(images), np.stack(labels_list).astype(np.int32)


def find_images_classification(root, dataset, split):
    """Find image names from VOC classification split file.

    Args:
        root: Root directory of VOC dataset
        dataset: Dataset name (e.g., 'VOC2007', 'VOC2012')
        split: Split name (e.g., 'train', 'val', 'test')

    Returns:
        List of image names (without newline characters)
    """
    root = Path(root)
    file_path = root / 'VOCdevkit' / dataset / 'ImageSets' / 'Main' / f'{split}.txt'

    print(f'[dataset] Reading {file_path}')
    with open(file_path, 'r') as f:
        images = [line.strip() for line in f if line.strip()]

    return images


def download_voc(root, version='2007'):
    """Download VOC dataset (2007 or 2012 version).

    Args:
        root: Root directory to download the dataset
        version: Dataset version, either '2007' or '2012' (default: '2007')
    """
    urls = {
        '2007': {
            'devkit': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar',
            'trainval': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
            'test_images': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
            'test_anno': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtestnoimgs_06-Nov-2007.tar'
        },
        '2012': {
            'devkit': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar',
            'trainval': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
            'test_images': 'http://pjreddie.com/media/files/VOC2012test.tar',
            'test_anno': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtestnoimgs_06-Nov-2012.tar',
        }
    }

    if version not in urls:
        raise ValueError(f"Unknown version: {version}. Use '2007' or '2012'.")

    root = Path(root)
    year = version
    urls_version = urls[version]
    path_devkit = root / 'VOCdevkit'
    path_images = root / 'VOCdevkit' / f'VOC{year}' / 'JPEGImages'
    tmpdir = root / 'tmp'

    # Create directories
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)

    # Download and extract devkit
    if not path_devkit.exists():
        if not tmpdir.exists():
            tmpdir.mkdir(parents=True, exist_ok=True)

        parts = urlparse(urls_version['devkit'])
        filename = Path(parts.path).name
        cached_file = tmpdir / filename

        if not cached_file.exists():
            print(f'Downloading: "{urls_version["devkit"]}" to {cached_file}\n')
            download_url(urls_version['devkit'], str(cached_file))

        print(f'[dataset] Extracting tar file {cached_file} to {root}')
        subprocess.run(
            f'tar -xf {cached_file}',
            shell=True,
            cwd=str(root)
        )
        print('[dataset] Done!')

    # Download and extract train/val images
    if not path_images.exists():
        key = 'trainval'
        parts = urlparse(urls_version[key])
        filename = Path(parts.path).name
        cached_file = tmpdir / filename

        if not cached_file.exists():
            print(f'Downloading: "{urls_version[key]}" to {cached_file}\n')
            download_url(urls_version[key], str(cached_file))

        print(f'[dataset] Extracting tar file {cached_file} to {root}')
        subprocess.run(
            f'tar -xf {cached_file}',
            shell=True,
            cwd=str(root)
        )
        print('[dataset] Done!')

    # Download test annotations
    test_anno = path_devkit / f'VOC{year}' / 'ImageSets' / 'Main' / 'aeroplane_test.txt'
    if not test_anno.exists():
        key = 'test_images'
        parts = urlparse(urls_version[key])
        filename = Path(parts.path).name
        cached_file = tmpdir / filename

        if not cached_file.exists():
            print(f'Downloading: "{urls_version[key]}" to {cached_file}\n')
            download_url(urls_version[key], str(cached_file))

        print(f'[dataset] Extracting tar file {cached_file} to {root}')
        subprocess.run(
            f'tar -xf {cached_file}',
            shell=True,
            cwd=str(root)
        )
        print('[dataset] Done!')

    # Download test images
    test_image = path_devkit / f'VOC{year}' / 'JPEGImages' / '000001.jpg'
    if not test_image.exists():
        key = 'test_anno'
        parts = urlparse(urls_version[key])
        filename = Path(parts.path).name
        cached_file = tmpdir / filename

        if not cached_file.exists():
            print(f'Downloading: "{urls_version[key]}" to {cached_file}\n')
            download_url(urls_version[key], str(cached_file))

        print(f'[dataset] Extracting tar file {cached_file} to {root}')
        subprocess.run(
            f'tar -xf {cached_file}',
            shell=True,
            cwd=str(root)
        )
        print('[dataset] Done!')


class Voc2007(data.Dataset):
    """PASCAL VOC 2007 dataset with noisy label support.

    Args:
        root: Root directory for dataset
        noise_type: Type of noise ('symmetric' or 'pairflip')
        noise_rate: Proportion of labels to corrupt (0.0 to 1.0)
        split_per: Proportion of trainval to use for training
        nb_classes: Number of classes (default: 20)
        num_workers: Number of workers for data loading
        noisy_val: Whether to apply noise to validation set (val).
                   clean_val is NEVER noisified.
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
        random_seed = 256

        # Initialize paths
        self.root = root
        root = Path(root)
        self.path_devkit = str(root / 'VOCdevkit')
        self.path_images = str(root / 'VOCdevkit' / 'VOC2007' / 'JPEGImages')

        # Download dataset if needed
        download_voc(self.root, version='2007')

        # Setup CSV file paths
        path_csv = root / 'files' / 'VOC2007'

        # Create CSV files if necessary
        for phase in ['trainval', 'test']:
            file_csv = path_csv / f'classification_{phase}.csv'
            if not file_csv.exists():
                path_csv.mkdir(parents=True, exist_ok=True)
                labeled_data = read_object_labels(self.root, 'VOC2007', phase)
                write_object_labels_csv(str(file_csv), labeled_data)

        self.classes = OBJECT_CATEGORIES

        # Load test data
        test_file_csv = path_csv / 'classification_test.csv'
        test_images, test_true_labels = read_object_labels_csv(str(test_file_csv))
        test_true_labels[test_true_labels == 0] = 1

        # Load trainval data
        trainval_file_csv = path_csv / 'classification_trainval.csv'
        trainval_images, trainval_true_labels = read_object_labels_csv(str(trainval_file_csv))
        trainval_true_labels[trainval_true_labels == 0] = 1

        path_images = Path(self.path_images)
        # Filter to only existing images
        test_indices = [
            (path_images / f'{img}.jpg').exists()
            for img in test_images
        ]
        test_images = test_images[test_indices]
        test_true_labels = test_true_labels[test_indices]
        assert len(test_images) == len(test_true_labels)

        trainval_indices = [
            (path_images / f'{img}.jpg').exists()
            for img in trainval_images
        ]
        trainval_images = trainval_images[trainval_indices]
        trainval_true_labels = trainval_true_labels[trainval_indices]
        assert len(trainval_images) == len(trainval_true_labels)
        print(f'Filtered image size: test: {len(test_images)}, trainval: {len(trainval_images)}')

        # Split trainval into train and val
        train_images, _, train_true_labels, val_images, _, val_true_labels = dataset_split(
            trainval_images, trainval_true_labels, trainval_true_labels,
            num_classes=len(self.classes)
        )

        # Generate noisy training labels
        train_labels = train_true_labels.copy()
        if noise_rate > 0:
            train_labels = generate_noisy_labels(
                train_true_labels,
                noise_type,
                noise_rate,
                nb_classes,
                random_seed
            )
        train_labels[train_labels == -1] = 0
        self.train_data = map_dataset_to_images(
            datasets.Dataset.from_dict({'image_file': train_images, 'labels': train_labels}),
            self.path_images,
            num_workers=num_workers
        )

        # Prepare validation sets (split into val and clean_val)
        val_labels = val_true_labels.copy()
        clean_val_labels = val_true_labels.copy()

        # Apply noise only to val (based on noisy_val), clean_val is NEVER noisified
        if noisy_val and noise_rate > 0:
            print('Noisify val set')
            val_labels = generate_noisy_labels(
                val_true_labels, noise_type, noise_rate, nb_classes, random_seed
            )
        val_labels[val_labels == -1] = 0
        clean_val_labels[clean_val_labels == -1] = 0

        # Single randomized split into val and clean_val
        val_images_split, clean_val_images, val_labels_split, clean_val_labels = train_test_split(
            val_images, val_labels, test_size=0.5, random_state=random_seed)

        self.val_data = map_dataset_to_images(
            datasets.Dataset.from_dict({'image_file': val_images_split, 'labels': val_labels_split}),
            self.path_images,
            num_workers=num_workers
        )

        self.clean_val_data = map_dataset_to_images(
            datasets.Dataset.from_dict({'image_file': clean_val_images, 'labels': clean_val_labels}),
            self.path_images,
            num_workers=num_workers
        )

        # Prepare test set
        test_true_labels[test_true_labels == -1] = 0
        self.test_data = map_dataset_to_images(
            datasets.Dataset.from_dict({'image_file': test_images, 'labels': test_true_labels}),
            self.path_images,
            num_workers=num_workers
        )

    def __len__(self):
        return len(self.train_data)

    def get_number_classes(self):
        return len(self.classes)


class Voc2012(Voc2007):
    """PASCAL VOC 2012 dataset with noisy label support.

    This class extends Voc2007 to handle the 2012 version of the dataset.

    Args:
        root: Root directory for dataset
        noise_type: Type of noise ('symmetric' or 'pairflip')
        noise_rate: Proportion of labels to corrupt (0.0 to 1.0)
        split_per: Proportion of trainval to use for training
        nb_classes: Number of classes (default: 20)
        num_workers: Number of workers for data loading
        noisy_val: Whether to apply noise to validation set (val).
                   clean_val is NEVER noisified.
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
        # Initialize paths for VOC2012
        self.root = root
        root = Path(root)
        self.path_devkit = str(root / 'VOCdevkit')
        self.path_images = str(root / 'VOCdevkit' / 'VOC2012' / 'JPEGImages')
        self.year = '2012'
        self.dataset_name = 'VOC2012'

        # Download dataset if needed
        download_voc(self.root, version='2012')

        # Setup CSV file paths
        path_csv = root / 'files' / 'VOC2012'

        # Create CSV files if necessary
        for phase in ['trainval', 'val']:
            file_csv = path_csv / ('classification_trainval.csv' if phase == 'trainval' else 'classification_test.csv')
            if not file_csv.exists():
                path_csv.mkdir(parents=True, exist_ok=True)
                labeled_data = read_object_labels(self.root, 'VOC2012', phase)
                write_object_labels_csv(str(file_csv), labeled_data)

        self.classes = OBJECT_CATEGORIES

        # Load test data
        test_file_csv = path_csv / 'classification_test.csv'
        test_images, test_true_labels = read_object_labels_csv(str(test_file_csv))
        test_true_labels[test_true_labels == 0] = 1

        # Load trainval data
        trainval_file_csv = path_csv / 'classification_trainval.csv'
        trainval_images, trainval_true_labels = read_object_labels_csv(str(trainval_file_csv))
        trainval_true_labels[trainval_true_labels == 0] = 1

        path_images = Path(self.path_images)
        # Filter to only existing images
        test_indices = [
            (path_images / f'{img}.jpg').exists()
            for img in test_images
        ]
        test_images = test_images[test_indices]
        test_true_labels = test_true_labels[test_indices]
        assert len(test_images) == len(test_true_labels)

        trainval_indices = [
            (path_images / f'{img}.jpg').exists()
            for img in trainval_images
        ]
        trainval_images = trainval_images[trainval_indices]
        trainval_true_labels = trainval_true_labels[trainval_indices]
        assert len(trainval_images) == len(trainval_true_labels)
        print(f'Filtered image size: test: {len(test_images)}, trainval: {len(trainval_images)}')

        # Split trainval into train and val
        train_images, _, train_true_labels, val_images, _, val_true_labels = dataset_split(
            trainval_images, trainval_true_labels, trainval_true_labels,
            num_classes=len(self.classes)
        )

        # Generate noisy training labels
        train_labels = train_true_labels.copy()
        if noise_rate > 0:
            train_labels = generate_noisy_labels(
                train_true_labels,
                noise_type,
                noise_rate,
                nb_classes,
                random_seed=256
            )
        train_labels[train_labels == -1] = 0
        self.train_data = map_dataset_to_images(
            datasets.Dataset.from_dict({'image_file': train_images, 'labels': train_labels}),
            self.path_images,
            num_workers=num_workers
        )

        # Prepare validation sets (split into val and clean_val)
        val_labels = val_true_labels.copy()
        clean_val_labels = val_true_labels.copy()

        # Apply noise only to val (based on noisy_val), clean_val is NEVER noisified
        if noisy_val and noise_rate > 0:
            print('Noisify val set')
            val_labels = generate_noisy_labels(
                val_true_labels, noise_type, noise_rate, nb_classes, random_seed=256
            )
        val_labels[val_labels == -1] = 0
        clean_val_labels[clean_val_labels == -1] = 0

        # Single randomized split into val and clean_val
        val_images_split, clean_val_images, val_labels_split, clean_val_labels = train_test_split(
            val_images, val_labels, test_size=0.5, random_state=256)

        self.val_data = map_dataset_to_images(
            datasets.Dataset.from_dict({'image_file': val_images_split, 'labels': val_labels_split}),
            self.path_images,
            num_workers=num_workers
        )

        self.clean_val_data = map_dataset_to_images(
            datasets.Dataset.from_dict({'image_file': clean_val_images, 'labels': clean_val_labels}),
            self.path_images,
            num_workers=num_workers
        )

        # Prepare test set
        test_true_labels[test_true_labels == -1] = 0
        self.test_data = map_dataset_to_images(
            datasets.Dataset.from_dict({'image_file': test_images, 'labels': test_true_labels}),
            self.path_images,
            num_workers=num_workers
        )
