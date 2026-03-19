import csv
import datasets 
import numpy as np
import os
import os.path
import pickle
import random
import tarfile
import torch
import torch.utils.data as data
import utils
from numpy.testing import assert_array_almost_equal
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse

from utils import *


object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']


ImageFile.LOAD_TRUCATED_IMAGES = True 


def dataset_split(train_images, train_labels, true_labels, split_per=0.9, random_seed=1, num_classes=10):
    """Split dataset into training and validation sets.
    
    Args:
        train_images: Array of training images
        train_labels: Array of training labels (noisy)
        true_labels: Array of ground truth labels
        split_per: Proportion of data for training (default: 0.9)
        random_seed: Random seed for reproducibility (default: 1)
        num_classes: Number of classes (unused parameter, kept for API compatibility)
    
    Returns:
        Tuple of (train_set, train_labels, train_true_labels, val_set, val_labels, val_true_labels)
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Get total number of samples
    num_samples = train_labels.shape[0]
    
    # Generate random permutation of indices
    shuffled_indices = np.random.permutation(num_samples)
    
    # Split indices based on ratio
    n_train = int(num_samples * split_per)
    train_indices = shuffled_indices[:n_train]
    val_indices = shuffled_indices[n_train:]
    
    # Split datasets using indices
    train_set = train_images[train_indices]
    val_set = train_images[val_indices]
    
    train_labels_split = train_labels[train_indices]
    val_labels = train_labels[val_indices]
    
    train_true_labels = true_labels[train_indices]
    val_true_labels = true_labels[val_indices]
    
    return train_set, train_labels_split, train_true_labels, val_set, val_labels, val_true_labels


def map_dataset_to_images(dataset, image_path, num_workers):   
    """Map dataset to load images in batch style.
    
    Args:
        dataset: datasets.Dataset with 'image_file' and 'labels' columns
    
    Returns:
        datasets.Dataset with 'data' (RGB images) and 'labels' columns
    """
    def load_image_batch(batch):
        """Load images for a batch of file paths.""" 
        return {
            'data': [
                Image.open(
                    os.path.join(image_path, f'{image_file}.jpg')
                ).convert('RGB')
                for image_file in iter(batch['image_file'])
            ]
        }
    
    return dataset.map(
        load_image_batch,
        remove_columns=['image_file'],
        num_proc=num_workers,
        batched=True,
        batch_size=128  # Process 64 images at a time
    ) 


def read_image_label(file):
    print('[dataset] read ' + file)
    data = dict()
    with open(file, 'r') as f:
        for line in f:
            tmp = line.split(' ')
            name = tmp[0]
            label = int(tmp[-1])
            data[name] = label
            
    return data


def read_object_labels(root, dataset, set):
    path_labels = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
    labeled_data = dict()
    num_classes = len(object_categories)

    for i in range(num_classes):
        file = os.path.join(path_labels, object_categories[i] + '_' + set + '.txt')
        data = read_image_label(file)

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
    # write a csv file
    print('[dataset] write file %s' % file)
    with open(file, 'w') as csvfile:
        fieldnames = ['name']
        fieldnames.extend(object_categories)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for (name, labels) in labeled_data.items():
            example = {'name': name}
            for i in range(20):
                example[fieldnames[i + 1]] = int(labels[i])
            writer.writerow(example)

    csvfile.close()


def read_object_labels_csv(file, header=True):
    images = []
    labels_list = []
    num_categories = 0
    print('[dataset] read', file)
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
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                labels = torch.from_numpy(labels)
                images.append(name)
                labels_list.append(labels)
            rownum += 1
    return np.stack(images), np.stack(labels_list).astype(np.int32)


def find_images_classification(root, dataset, set):
    """Find image names from VOC classification split file.
    
    Args:
        root: Root directory of VOC dataset
        dataset: Dataset name (e.g., 'VOC2007', 'VOC2012')
        set: Split name (e.g., 'train', 'val', 'test')
    
    Returns:
        List of image names (without newline characters)
    """
    file_path = os.path.join(
        root, 'VOCdevkit', dataset, 'ImageSets', 'Main',  f'{set}.txt'
    ) 
    
    print(f'[dataset] Reading {file_path}')
    with open(file_path, 'r') as f:
        # Read all lines and strip whitespace/newlines efficiently
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
            'trainval_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
            'test_images_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
            'test_anno_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtestnoimgs_06-Nov-2007.tar'
        }, 
        '2012': {
            'devkit': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar',
            'trainval_2012': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
            'test_images_2012': 'http://pjreddie.com/media/files/VOC2012test.tar',
            'test_anno_2012': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtestnoimgs_06-Nov-2012.tar',
        }
    }
 

    # Select URLs based on version
    if version == '2007': 
        year = '2007'
    elif version == '2012': 
        year = '2012'
    else:
        raise ValueError(f"Unknown version: {version}. Use '2007' or '2012'.")
    
    urls_version = urls[version]
    path_devkit = os.path.join(root, 'VOCdevkit')
    path_images = os.path.join(root, 'VOCdevkit', f'VOC{year}', 'JPEGImages')
    tmpdir = os.path.join(root, 'tmp')

    # create directory
    if not os.path.exists(root):
        os.makedirs(root)

    if not os.path.exists(path_devkit):

        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)

        parts = urlparse(urls_version['devkit'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(urls_version['devkit'], cached_file))
            utils.download_url(urls_version['devkit'], cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')

    # train/val images/annotations
    if not os.path.exists(path_images):
        # download train/val images/annotations
        parts = urlparse(urls_version['trainval_2007' if version == '2007' else 'trainval_2012'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(
                    urls_version['trainval_2007' if version == '2007' else 'trainval_2012'], cached_file
                )
            )
            utils.download_url(
                urls_version['trainval_2007' if version == '2007' else 'trainval_2012'], cached_file
            )

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')

    # test annotations
    test_anno = os.path.join(path_devkit, f'VOC{year}/ImageSets/Main/aeroplane_test.txt')
    if not os.path.exists(test_anno):

        # download test annotations
        parts = urlparse(urls_version['test_images_2007' if version == '2007' else 'test_images_2012'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            print(
                'Downloading: "{}" to {}\n'.format(
                    urls_version['test_images_2007' if version == '2007' else 'test_images_2012'], 
                    cached_file
                )
            )
            utils.download_url(
                urls_version['test_images_2007' if version == '2007' else 'test_images_2012'], 
                cached_file
            )

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')

    # test images
    test_image = os.path.join(path_devkit, f'VOC{year}/JPEGImages/000001.jpg')
    if not os.path.exists(test_image):

        # download test images
        parts = urlparse(urls_version['test_anno_2007' if version == '2007' else 'test_anno_2012'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(urls_version['test_anno_2007' if version == '2007' else 'test_anno_2012'], cached_file))
            utils.download_url(urls_version['test_anno_2007' if version == '2007' else 'test_anno_2012'], cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')


class Voc2007(data.Dataset):
    """VOC2007 dataset with noisy label support."""
    
    def __init__(
        self, 
        root,  
        noise_type='symmetric', 
        noise_rate=0.3, 
        split_per=0.9, 
        nb_classes=20,
        num_workers=4,
        noisy_val=False,
        version='2007'  # Added version parameter for flexibility
    ):
        # Set random seed for reproducibility
        random_seed = 1 
        
        # Store version and initialize paths
        self.version = version
        self.root = root
        self.path_devkit = os.path.join(root, 'VOCdevkit')
        self.path_images = os.path.join(root, 'VOCdevkit', f'VOC{version}', 'JPEGImages')
   
        # Download dataset if not exists
        download_voc(self.root, version=version)

        # Setup CSV file paths
        path_csv = os.path.join(self.root, 'files', f'VOC{version}')
        
        # Create CSV files if necessary
        for phase in self._get_csv_phases():
            file_csv = os.path.join(path_csv, f'classification_{phase}.csv')
            if not os.path.exists(file_csv):
                # Create directory if necessary
                if not os.path.exists(path_csv):
                    os.makedirs(path_csv)
                    
                # Generate and write CSV file
                labeled_data = read_object_labels(self.root, f'VOC{version}', phase)
                write_object_labels_csv(file_csv, labeled_data)

        self.classes = object_categories

        # Load test data
        test_file_csv = os.path.join(path_csv, self._get_test_filename())
        test_images, test_true_labels = read_object_labels_csv(test_file_csv)
        test_true_labels[test_true_labels==0] = 1
        
        # Load trainval data
        trainval_file_csv = os.path.join(path_csv, 'classification_trainval.csv')
        trainval_images, trainval_true_labels = read_object_labels_csv(trainval_file_csv)
        trainval_true_labels[trainval_true_labels==0] = 1

        # Filter images that only exist
        test_indices = [
            os.path.exists(os.path.join(self.path_images, f"{img}.jpg")) 
            for img in iter(test_images)
        ]
        test_images = test_images[test_indices]
        test_true_labels = test_true_labels[test_indices]
        assert len(test_images) == len(test_true_labels)
        
        trainval_indices = [
            os.path.exists(os.path.join(self.path_images, f"{img}.jpg")) 
            for img in iter(trainval_images)
        ] 
        trainval_images = trainval_images[trainval_indices]
        trainval_true_labels = trainval_true_labels[trainval_indices]
        assert len(trainval_images) == len(trainval_true_labels)
        print(f'filtered image size: test: {len(test_images)}, trainval: {len(trainval_images)}')

        train_images, _, train_true_labels, val_images, _, val_true_labels = dataset_split(
            trainval_images, trainval_true_labels, trainval_true_labels, 
            num_classes=len(self.classes)
        )

        train_labels = train_true_labels
        if noise_rate > 0:
            train_labels = generate_noisy_labels(
                train_true_labels, 
                noise_type, 
                noise_rate, 
                nb_classes, 
                random_seed
            ) 
            
        # Change -1 if any to 0
        train_labels[train_labels==-1] = 0 
        self.train_data = map_dataset_to_images(
            datasets.Dataset.from_dict({'image_file': train_images, 'labels': train_labels}),
            self.path_images,
            num_workers=num_workers
        ) 
 
        val_true_labels[val_true_labels==-1] = 0
        val_images0, val_images1, val_labels0, val_labels1 = train_test_split(
            val_images, val_true_labels, test_size=0.5, random_state=42)
         
        if noisy_val and noise_rate > 0:
            print('Noisify val set')
            val_labels0 = generate_noisy_labels(
                val_labels0, noise_type, noise_rate, nb_classes, random_seed
            )

        # noisy val data
        self.val_data0 = map_dataset_to_images(
            datasets.Dataset.from_dict({'image_file': val_images0, 'labels': val_labels0}),
            self.path_images,
            num_workers=num_workers
        )

        # clean val data 
        self.val_data1 = map_dataset_to_images(
            datasets.Dataset.from_dict({'image_file': val_images1, 'labels': val_labels1}),
            self.path_images,
            num_workers=num_workers
        ) 

        test_true_labels[test_true_labels==-1] = 0
        self.test_data = map_dataset_to_images(
            datasets.Dataset.from_dict({'image_file': test_images, 'labels': test_true_labels}),
            self.path_images,
            num_workers=num_workers
        )
    
    def _get_csv_phases(self):
        """Get CSV file phases for this dataset version.
        
        Returns:
            List of phase names (e.g., ['trainval', 'test'])
        """
        return ['trainval', 'test']
    
    def _get_test_filename(self):
        """Get test filename for this dataset version.
        
        Returns:
            Filename string
        """
        return 'classification_test.csv'
  
    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)


class Voc2012(Voc2007):
    """VOC2012 dataset as a subclass of Voc2007.
    
    Inherits all functionality from Voc2007, with specific overrides for:
    - Dataset version (2012)
    - CSV phases (uses 'val' instead of 'test')
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
        # Call parent class constructor with version='2012'
        super().__init__(
            root=root,
            noise_type=noise_type,
            noise_rate=noise_rate,
            split_per=split_per,
            nb_classes=nb_classes,
            num_workers=num_workers,
            noisy_val=noisy_val,
            version='2012'
        )
      
def generate_noisy_labels(labels, noise_type, noise_rate, nb_classes, random_seed):
    N, nc = labels.shape
    labels[labels==0] = 1
    labels[labels==-1] = 0
    
    if noise_type == 'symmetric':
        noisy_labels, _, _ = noisify_multiclass_symmetric(
            labels, noise_rate, random_state=random_seed, nb_classes=nb_classes
        )
    else:
        noisy_labels, _, _ = noisify_pairflip(
            labels, noise_rate, random_state=random_seed, nb_classes=nb_classes
        ) 
    return noisy_labels


def multiclass_noisify(y, P, random_state=None):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()
    m, l = y.shape[0], y.shape[1]
    new_y = np.ones((m, l))
    #print(m, l)
    noise_or_not = 0.
    total_label = 0.
    for i in range(m):
        label = np.array(y[i], dtype='int')      
        idx_label = np.where(label==1)[0]
        iteration = 0
        idx_label_ = np.zeros((100000, ))
        while int(idx_label_.shape[0]) != int(idx_label.shape[0]):
            new_a = np.zeros((1, l))
            iteration += 1
            for idx in range(int(idx_label.shape[0])):    
                # print(idx)
                k = idx_label[idx]
                flipped = np.random.multinomial(1, P[k, :], 1)[0]
                flipped = flipped.reshape(1, l)
                new_a += flipped
                new_a = np.array(new_a, dtype='int')
            idx_label_ = np.where(new_a==1)[0]
            if int(idx_label_.shape[0]) == int(idx_label.shape[0]):
                break
        new_y[i, :] = new_a[0, :]
        b = np.sum(new_a.astype('int') != label.astype('int')) / 2
        noise_or_not += b
        total_label += idx_label.shape[0]
    return new_y


def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=20):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        y_train_noisy = multiclass_noisify(
            y_train, P=P,
            random_state=random_state
        )
        actual_noise = np.sum(np.abs(y_train_noisy-y_train)) / np.sum(y_train) * 0.5 
    else:
        y_train_noisy = y_train
        actual_noise = 0.
    return y_train_noisy, actual_noise, P


def noisify_pairflip(y_train, noise, random_state=None, nb_classes=20):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = np.sum(np.abs(y_train_noisy-y_train)) / np.sum(y_train) * 0.5 
    else:
        y_train_noisy = y_train
        actual_noise = 0.
        
    return y_train_noisy, actual_noise, P