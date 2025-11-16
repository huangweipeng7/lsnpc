import datasets
import json
import numpy as np
import os
import pandas as pd
import pickle
import random
import subprocess
import sys
import torch
import pickle
import torch.utils.data as data
from numpy.testing import assert_array_almost_equal
from PIL import Image
from sklearn.model_selection import train_test_split

from utils import *



class CIFAR10(data.Dataset):
    """FashionMNIST dataset from https://www.cs.toronto.edu/~kriz/cifar.html
       Download & unzip the dataset into [root]/cifar10
        data_batch_1 ~ data_batch_5: training data
        test_batch: test data
    
    """
    def __init__(self, 
        root,  
        noise_type = 'symmetric', 
        noise_rate=0.0, 
        split_per=0.9, 
        nb_classes=10, 
        num_workers=4,
        noisy_val=False
    ):
        random_seed = 1 

        self.root = root 
        self.num_classes = nb_classes

        df_dict = self.get_anno()
        
        # --- Setup training set ---
        train_data = df_dict['train']

        if noise_rate > 0:
            print('Creating noisy labels for training set...')
            train_data['labels'] = generate_noisy_labels(
                train_data['labels'], noise_type, noise_rate, nb_classes, random_seed
            )
        else:
            train_data['labels'] = np.array(train_data['labels'])
            train_data['labels'] = train_data['labels'].tolist()

        self.train_data = datasets.Dataset.from_dict(
            train_data
        ).map(
            lambda batch: {
                'data': Image.fromarray(np.array(batch['image_array'], dtype=np.uint8)
                                        .reshape(3, 32, 32)
                                        .transpose(1, 2, 0), 
                                        mode='RGB')
            }, 
            remove_columns=['image_array'],
            num_proc=num_workers
        ) 

        # --- Setup validation set part 1 ---
        val_data0 = df_dict['val0']
        if noisy_val:
            print('Noisify val set')
            val_data0['labels'] = generate_noisy_labels(
                val_data0['labels'], noise_type, noise_rate, nb_classes, random_seed
            )
        val_data0['labels'] = np.array(val_data0['labels'])
        val_data0['labels'] = val_data0['labels'].tolist()
        self.val_data0 = datasets.Dataset.from_dict(
            val_data0
        ).map(
            lambda batch: {
                'data': Image.fromarray(np.array(batch['image_array'], dtype=np.uint8)
                                        .reshape(3, 32, 32)
                                        .transpose(1, 2, 0), 
                                        mode='RGB')
            }, 
            remove_columns=['image_array'],
            num_proc=num_workers
        )

        # --- Setup validation set part 2 ---
        val_data1 = df_dict['val1']
        if noisy_val:
            print('Noisify val set')
            val_data1['labels'] = generate_noisy_labels(
                val_data1['labels'], noise_type, noise_rate, nb_classes, random_seed
            )
        val_data1['labels'] = np.array(val_data1['labels'])
        val_data1['labels'] = val_data1['labels'].tolist()
        self.val_data1 = datasets.Dataset.from_dict(
            val_data1
        ).map(
            lambda batch: {
                'data': Image.fromarray(np.array(batch['image_array'], dtype=np.uint8)
                                        .reshape(3, 32, 32)
                                        .transpose(1, 2, 0), 
                                        mode='RGB')
            }, 
            remove_columns=['image_array'],
            num_proc=num_workers
        )

        # --- Setup test set ---
        test_data = df_dict['test']
        test_data['labels'] = np.array(test_data['labels'])
        # test_data['labels'][test_data['labels'] == -1] = 0
        test_data['labels'] = test_data['labels'].tolist()
        self.test_data = datasets.Dataset.from_dict(
            test_data
        ).map(
            lambda batch: {
                'data': Image.fromarray(np.array(batch['image_array'], dtype=np.uint8)
                                        .reshape(3, 32, 32)
                                        .transpose(1, 2, 0), 
                                        mode='RGB')
            }, 
            remove_columns=['image_array'],
            num_proc=num_workers
        )


    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    

    def get_anno(self): 
        """ Return the dictionary of dataframe for train/val/test splits
            Each dataframe contains `image_path` and `labels` (list of multi-labels)
        """
        # print('Preparing NUS-WIDE annotations...')
        anno_path = os.path.join(
            self.root, 'cifar10'
        )

        df_dict = {}

        # data1 = self.unpickle(os.path.join(anno_path, 'data_batch_1'))

        # print(data1.keys())
        # from PIL import Image
        # test_img = np.array(data1[b'data'][0]).reshape(3, 32, 32).transpose(1, 2, 0)
        # pil_image = Image.fromarray(test_img, mode='RGB')
        # pil_image.save('test_image.png')

        # # Then open with PIL
        
        # opened_image = Image.open('test_image.png')
        # opened_image.show()

        # print(data1[b'labels'][:5], max(data1[b'labels']), min(data1[b'labels']))
        # print(np.eye(self.num_classes)[data1[b'labels'][:5]])

        # ----- Train and val split ------------------------------------------------------
        train_image_arrays = []
        train_labels = []
        for train_batch_idx in range(1, 6):
            data_batch = self.unpickle(
                os.path.join(anno_path, f'data_batch_{train_batch_idx}')
            )
            train_image_arrays.append(data_batch[b'data'])
            train_labels.extend(data_batch[b'labels'])
        
        train_image_arrays = np.vstack(train_image_arrays)
        train_labels = np.eye(self.num_classes)[train_labels]

        print(train_image_arrays.shape)
        print(train_labels.shape)

        assert np.max(train_labels) == 1
        assert np.min(train_labels) == 0

        # stratify_keys = [''.join(map(str, label)) for label in train_labels]

        train_array, val_array, train_labs, val_labs = train_test_split(
            train_image_arrays, 
            train_labels,
            test_size=0.2,  
            random_state=42
        )

        # Further split validation into val0 and val1
        val_array0, val_array1, val_labs0, val_labs1 = train_test_split(
            val_array, 
            val_labs,
            test_size=0.5,
            random_state=42
        )

        # Prepare datasets
        df_dict['train'] = {
            'image_array': train_array.astype(np.uint8).tolist(),
            'labels': train_labs.tolist()
        }
        
        df_dict['val0'] = {
            'image_array': val_array0.astype(np.uint8).tolist(),
            'labels': val_labs0.tolist()
        }
        
        df_dict['val1'] = {
            'image_array': val_array1.astype(np.uint8).tolist(),
            'labels': val_labs1.tolist()
        }


        # ----- Test set -----------------------------------------------------------
        test_batch = self.unpickle(
            os.path.join(anno_path, f'test_batch')
        )
        test_image_arrays = test_batch[b'data']
        test_labels = test_batch[b'labels']

        # test_image_arrays = test_image_array
        test_labels = np.eye(self.num_classes)[test_labels]

        print(test_image_arrays.shape)
        print(test_labels.shape)

        df_dict['test'] = {
            'image_array': test_image_arrays.astype(np.uint8).tolist(),
            'labels': test_labels.tolist()
        }

        return df_dict


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item, target = self.img_list[index], self.labels[index]
        return self.get(item), target

    def get_number_classes(self):
        return self.num_classes


def generate_noisy_labels(labels, noise_type, noise_rate, nb_classes, random_seed):

    labels = np.array(labels)
    
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
    #print(m, l)
    noise_or_not = 0.
    total_label = 0.
    for i in range(m):
        label = np.array(y[i], dtype='int')      
        #print(f'i:{i}, label:{label}')
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
        #print(f'new_a: {new_a[0,:]}')
        new_y[i, :] = new_a[0, :]
        b = np.sum(new_a.astype('int') != label.astype('int')) / 2
        #print(f'm:{m}, b:{b}')
        noise_or_not += b
        total_label += idx_label.shape[0]
    return new_y


def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=8):
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

        #print('P', P)

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = np.sum(np.abs(y_train_noisy-y_train)) / np.sum(y_train) * 0.5
        #print(f'Sum of diff: {np.sum(np.abs(y_train_noisy-y_train))}')
        #print(f'Sum of y_train: {np.sum(y_train)}')
        print('Actual noise %.2f' % actual_noise)
        print(P)
    else:
        y_train_noisy = y_train
        actual_noise = 0.
    return y_train_noisy, actual_noise, P


def noisify_pairflip(y_train, noise, random_state=None, nb_classes=8):
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


if __name__ == '__main__':
    cifar10 = CIFAR10(root='data')


