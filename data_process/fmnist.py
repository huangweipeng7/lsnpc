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
from datasets import load_dataset

from utils import *



class FMNIST(data.Dataset):
    """ FashionMNIST dataset from HuggingFace Datasets library """

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

        ds = load_dataset('zalando-datasets/fashion_mnist')
        ds = ds.rename_columns(
            {
                'label': 'labels',
                'image': 'data'
            }
        )
        
        # Change labels to one-hot encoding
        one_hot_labels = np.eye(self.num_classes)
        ds = ds.map(
            lambda batch: {
                'labels': one_hot_labels[batch['labels']],
                'data': batch['data'].convert("RGB")
            },
            num_proc=num_workers
        )
        print(ds['train'][0])

        # Keep test as it is
        self.test_data = ds['test']

        # ----- Split train into train, val0, val1 -----
        train_ds = ds['train']
        train_ds = train_ds.shuffle(seed=random_seed)

        split_20 = train_ds.train_test_split(test_size=0.2, seed=random_seed)
        remaining_train_ds = split_20['train']
        temp_20 = split_20['test']

        # Split that 20% into two equal 10% splits
        val_split = temp_20.train_test_split(test_size=0.5, seed=random_seed)

        val0 = val_split["train"]   # 10%
        val1 = val_split["test"]    # 10%

        print("Training labels:", np.where(np.array(remaining_train_ds['labels']) == 1)[1][:10])
        print("Val0 labels:", np.where(np.array(val0['labels']) == 1)[1][:10])
        print("Val1 labels:", np.where(np.array(val1['labels']) == 1)[1][:10])

        self.train_data = remaining_train_ds
        self.val_data0 = val0
        self.val_data1 = val1

        # Update noisy labels
        if noise_rate > 0:
            print('Creating noisy labels for training set...')
            train_labels_with_noise = generate_noisy_labels(
                self.train_data['labels'], 
                noise_type, 
                noise_rate, 
                nb_classes, 
                random_seed
            ).tolist()
            self.train_data = self.train_data.remove_columns("labels")
            self.train_data = self.train_data.add_column("labels", train_labels_with_noise)

            print('Noisy training labels:', np.where(np.array(self.train_data['labels']) == 1)[1][:10])

            if noisy_val:
                print('Creating noisy labels for validation set...')
                val0_labels_with_noise = generate_noisy_labels(
                    self.val_data0['labels'], 
                    noise_type, 
                    noise_rate, 
                    nb_classes, 
                    random_seed
                ).tolist()
                val1_labels_with_noise = generate_noisy_labels(
                    self.val_data1['labels'], 
                    noise_type, 
                    noise_rate, 
                    nb_classes, 
                    random_seed
                ).tolist()
                self.val_data0 = self.val_data0.remove_columns("labels")
                self.val_data0 = self.val_data0.add_column("labels", val0_labels_with_noise)
                self.val_data1 = self.val_data1.remove_columns("labels")
                self.val_data1 = self.val_data1.add_column("labels", val1_labels_with_noise)

                print('Noisy validation labels (val0):', np.where(np.array(self.val_data0['labels']) == 1)[1][:10])
                print('Noisy validation labels (val1):', np.where(np.array(self.val_data1['labels']) == 1)[1][:10])


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
    fmnist = FMNIST(root='data')


