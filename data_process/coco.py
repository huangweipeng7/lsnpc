import itertools
import datasets
import json
import numpy as np
import os
import pickle
import random
import subprocess
import sys
import torch
import torch.utils.data as data

from numpy.testing import assert_array_almost_equal
from PIL import Image
from utils import *


urls = {'train_img':'http://images.cocodataset.org/zips/train2014.zip',
        'val_img' : 'http://images.cocodataset.org/zips/val2014.zip',
        'annotations':'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'}


def download_coco2014(root, phase='train'):
    if not os.path.exists(root):
        os.makedirs(root)
    tmpdir = os.path.join(root, 'tmp/')
    data = os.path.join(root, '')

    if not os.path.exists(data):
        os.makedirs(data)
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    if phase == 'train':
        filename = 'train2014.zip'
    elif phase == 'val':
        filename = 'val2014.zip'
    cached_file = os.path.join(tmpdir, filename)
    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(urls[phase + '_img'], cached_file))
        os.chdir(tmpdir)
        subprocess.call('wget ' + urls[phase + '_img'], shell=True)
        os.chdir('../')
    # extract file
    img_data = os.path.join(data, filename.split('.')[0])
    if not os.path.exists(img_data):
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=data))
        command = 'unzip {} -d {}'.format(cached_file, data)
        os.system(command)
        command = f'rm {cached_file}'
        os.system(command)
    print('[dataset] Done!')

    # train/val images/annotations
    cached_file = os.path.join(tmpdir, 'annotations_trainval2014.zip')
    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(urls['annotations'], cached_file))
        os.chdir(tmpdir)
        subprocess.Popen('wget ' + urls['annotations'], shell=True)
        os.chdir(root)
    annotations_data = os.path.join(data, 'annotations')
    if not os.path.exists(annotations_data):
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=data))
        command = 'unzip {} -d {}'.format(cached_file, data)
        os.system(command)
    print('[annotation] Done!')

    anno = os.path.join(data, '{}_anno.json'.format(phase))
    img_id = {}
    annotations_id = {}
    if not os.path.exists(anno):
        annotations_file = json.load(open(os.path.join(annotations_data, 'instances_{}2014.json'.format(phase))))
        annotations = annotations_file['annotations']
        category = annotations_file['categories']
        category_id = {}
        for cat in category:
            category_id[cat['id']] = cat['name']
        cat2idx = categoty_to_idx(sorted(category_id.values()))
        images = annotations_file['images']
        for annotation in annotations:
            if annotation['image_id'] not in annotations_id:
                annotations_id[annotation['image_id']] = set()
            annotations_id[annotation['image_id']].add(cat2idx[category_id[annotation['category_id']]])
        for img in images:
            if img['id'] not in annotations_id:
                continue
            if img['id'] not in img_id:
                img_id[img['id']] = {}
            img_id[img['id']]['file_name'] = img['file_name']
            img_id[img['id']]['labels'] = list(annotations_id[img['id']])
        anno_list = []
        for k, v in img_id.items():
            anno_list.append(v)
        json.dump(anno_list, open(anno, 'w'))
        anno_list_2 = []
        for k, v in img_id.items():
            anno_list_2.append(v['labels'])
        anno = os.path.join(data, '{}_anno2.json'.format(phase))
        json.dump(anno_list_2, open(anno, 'w'))
        if not os.path.exists(os.path.join(data, 'category.json')):
            json.dump(cat2idx, open(os.path.join(data, 'category.json'), 'w'))
        del img_id
        del anno_list
        del anno_list_2
        del images
        del annotations_id
        del annotations
        del category
        del category_id
    print('[json] Done!')


def categoty_to_idx(category):
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx


class COCO2014(data.Dataset):
    def __init__(
        self, 
        root,  
        phase='train', 
        noise_type='symmetric', 
        noise_rate=0.3, 
        split_per=0.9, 
        nb_classes =80, 
        num_workers=4,
        noisy_val=False
    ):
        random_seed = 1

        self.root = root 
        for phase in ['train', 'val',]:
            download_coco2014(root, phase)
        
        with open(os.path.join(self.root, 'category.json')) as f:
            self.cat2idx = json.load(f)

        df_dict = self.get_anno()
        self.num_classes = len(self.cat2idx) 

        train_data = df_dict['train']
        train_data['labels'] = generate_noisy_labels(
            train_data['labels'], noise_type, noise_rate, nb_classes, random_seed
        )
        print(train_data['image_path'][:5])
        print(train_data['labels'][:5])
        self.train_data = datasets.Dataset.from_dict(
            train_data
        ).map(
            lambda batch: {
                'data': Image.open(
                    os.path.join(self.root, 'train2014', batch['image_path']['file_name'])
                ).convert('RGB')
            }, 
            remove_columns=['image_path'],
            num_proc=num_workers
        ) 

        val_data = df_dict['val']

        # divide val data into two parts for validation and test
        # save indicies for val for reproducibility in pickle file
        # load from pickle file if exists, otherwise create one
        VAL_SPLIT_SIZE = 10_000
        val_size = len(val_data['labels'])
        indice_file = './data_process/coco_val_indices.pkl'
        if os.path.exists(indice_file):
            print(f'Using exising validation indices')
            with open(indice_file, 'rb') as file:
                val_indices = pickle.load(file)
        else:
            print(f'Creating indices pickle file for validation split')
            val_indices = random.sample(range(0, val_size-1), VAL_SPLIT_SIZE)
            with open(indice_file, 'wb') as file:
                pickle.dump(val_indices, file)
        
        test_data = dict()
        test_data['image_path'] = [val_data['image_path'][i] for i in range(val_size) if i not in val_indices]
        test_data['labels'] = [val_data['labels'][i] for i in range(val_size) if i not in val_indices]
        val_data['image_path'] = [val_data['image_path'][i] for i in val_indices]
        val_data['labels'] = [val_data['labels'][i] for i in val_indices]
        print(len(val_data['image_path']), len(test_data['image_path']))
        
        
        #TEST_NUM = 1000 # small num to test the pipeline
        #val_data['image_path'] = val_data['image_path'][:TEST_NUM]
        #val_data['labels'] = val_data['labels'][:TEST_NUM]
        #print(val_data['labels'][0])

        if noisy_val:
            print('Noisify val set')
            val_data['labels'] = generate_noisy_labels(
                val_data['labels'], noise_type, noise_rate, nb_classes, random_seed
            )
        val_data['labels'] = np.array(val_data['labels'])
        val_data['labels'][val_data['labels'] == -1] = 0
        val_data['labels'] = val_data['labels'].tolist()
        print('Check labels')
        print(val_data['labels'][:2])
        self.val_data = datasets.Dataset.from_dict(
            val_data
        ).map(
            lambda batch: {
                'data': Image.open(
                    os.path.join(self.root, 'val2014', batch['image_path']['file_name'])
                ).convert('RGB')
            }, 
            remove_columns=['image_path'],
            num_proc=num_workers
        )

        #test_data = df_dict['test']
        test_data['labels'] = np.array(test_data['labels'])
        test_data['labels'][test_data['labels'] == -1] = 0
        test_data['labels'] = test_data['labels'].tolist()
        self.test_data = datasets.Dataset.from_dict(
            test_data
        ).map(
            lambda batch: {
                'data': Image.open(
                    os.path.join(self.root, 'val2014', batch['image_path']['file_name'])
                ).convert('RGB') 
            },
            remove_columns=['image_path'],
            num_proc=num_workers
        )
        #self.test_data = self.val_data
        #self.train_data = self.val_data

    def get_anno(self):
        df_dict = {}
        #df = pd.read_csv(anno_path)
        for phase in ['train', 'val',]: 
            list_path = os.path.join(self.root, f'{phase}_anno.json')
            tmp_img_list = json.load(open(list_path, 'r'))
            tmp_true_labels = self.get_true_labels(phase)

            df_dict[f'{phase}'] = {
                'image_path': tmp_img_list, 'labels': tmp_true_labels
            }
        return df_dict

    def get_true_labels(self, phase):
        list_path = os.path.join(self.root, f'{phase}_anno2.json')
        labels = json.load(open(list_path, 'r'))
        true_labels = np.zeros((len(labels), len(self.cat2idx)))-1
        for i, label in enumerate(labels):
            true_labels[i, label] = 1
        return true_labels.astype(np.int32)
       
    def __len__(self):
        return len(self.img_list)

    def get_number_classes(self):
        return self.num_classes 


def generate_noisy_labels(labels, noise_type, noise_rate, nb_classes, random_seed):
    N, nc = labels.shape
    labels[labels==0] = 1
    labels[labels==-1] = 0
    
    if noise_type == 'symmetric':
        noisy_labels, _, _ = noisify_multiclass_symmetric(labels, noise_rate, random_state=random_seed, nb_classes=nb_classes)
    else:
        noisy_labels, _, _ = noisify_pairflip(labels, noise_rate, random_state=random_seed, nb_classes=nb_classes) 
    
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
    print(m, l)
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


def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=80):
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

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = np.sum(np.abs(y_train_noisy-y_train)) / np.sum(y_train) * 0.5
        print('Actual noise %.2f' % actual_noise)
        print(P)
    else:
        y_train_noisy = y_train
        actual_noise = 0.
    return y_train_noisy, actual_noise, P


def noisify_pairflip(y_train, noise, random_state=None, nb_classes=80):
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
        print('Actual noise %.2f' % actual_noise)
        print(P)
    else:
        y_train_noisy = y_train
        actual_noise = 0.
        
    return y_train_noisy, actual_noise, P

