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
import torch.utils.data as data
from numpy.testing import assert_array_almost_equal
from PIL import Image
from sklearn.model_selection import train_test_split

from utils import *


def category_to_idx(category):
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx


class Tomato(data.Dataset):
    """Tomato dataset
        The zip file needs to be unziped into [root]/Tomato folder with following file and subfolders
        The [root] is the root folder we set with our argument
        .
        /train
        /test
        /val
        anno.csv (the csv file from Variant-b(MultiLabel Classification)
    """
    def __init__(self, 
        root,  
        noise_type = 'symmetric', 
        noise_rate=0.3, 
        split_per=0.9, 
        nb_classes=8, 
        num_workers=4,
        noisy_val=False
    ):
        random_seed = 1 

        self.root = root 
        self.cat2idx = category_to_idx([
            'Early blight', 
            'Healthy', 
            'Late blight', 
            'Leaf Miner',
            'Magnesium Deficiency',
            'Nitrogen Deficiency',
            'Pottassium Deficiency',
            'Spotted Wilt Virus'
        ])
        df_dict = self.get_anno()
        self.num_classes = len(self.cat2idx) 

        train_data = df_dict['train']

        if noise_rate > 0:
            train_data['labels'] = generate_noisy_labels(
                train_data['labels'], noise_type, noise_rate, nb_classes, random_seed
            )
        else:
            train_data['labels'] = np.array(train_data['labels'])
            train_data['labels'][train_data['labels'] == -1] = 0
            train_data['labels'] = train_data['labels'].tolist()
        self.train_data = datasets.Dataset.from_dict(
            train_data
        ).map(
            lambda batch: {
                'data': Image.open(
                    os.path.join(self.root, 'tomato', 'train', batch['image_path'])
                ).convert('RGB')
            }, 
            remove_columns=['image_path'],
            num_proc=num_workers
        ) 

        val_data0 = df_dict['val0']
        if noisy_val:
            print('Noisify val set')
            val_data0['labels'] = generate_noisy_labels(
                val_data0['labels'], noise_type, noise_rate, nb_classes, random_seed
            )
        val_data0['labels'] = np.array(val_data0['labels'])
        val_data0['labels'][val_data0['labels'] == -1] = 0
        val_data0['labels'] = val_data0['labels'].tolist()
        self.val_data0 = datasets.Dataset.from_dict(
            val_data0
        ).map(
            lambda batch: {
                'data': Image.open(
                    os.path.join(self.root, 'tomato', 'val', batch['image_path'])
                ).convert('RGB')
            }, 
            remove_columns=['image_path'],
            num_proc=num_workers
        )

        val_data1 = df_dict['val1']
        if noisy_val:
            print('Noisify val set')
            val_data1['labels'] = generate_noisy_labels(
                val_data1['labels'], noise_type, noise_rate, nb_classes, random_seed
            )
        val_data1['labels'] = np.array(val_data1['labels'])
        val_data1['labels'][val_data1['labels'] == -1] = 0
        val_data1['labels'] = val_data1['labels'].tolist()
        self.val_data1 = datasets.Dataset.from_dict(
            val_data1
        ).map(
            lambda batch: {
                'data': Image.open(
                    os.path.join(self.root, 'tomato', 'val', batch['image_path'])
                ).convert('RGB')
            }, 
            remove_columns=['image_path'],
            num_proc=num_workers
        )

        test_data = df_dict['test']
        test_data['labels'] = np.array(test_data['labels'])
        test_data['labels'][test_data['labels'] == -1] = 0
        test_data['labels'] = test_data['labels'].tolist()
        self.test_data = datasets.Dataset.from_dict(
            test_data
        ).map(
            lambda batch: {
                'data': Image.open(
                    os.path.join(self.root, 'tomato', 'test', batch['image_path'])
                ).convert('RGB') 
            },
            remove_columns=['image_path'],
            num_proc=num_workers
        )


    def get_anno(self): 
        anno_path = os.path.join(
            self.root, 'tomato', 'anno.csv'
        )

        df_dict = {}
        df = pd.read_csv(anno_path)
        #print(df)
        for phase in ['train', 'val', 'test']: 
            tmp_df = df[df['dataset']==phase].copy()
            tmp_img_list = tmp_df['filename'].values
            tmp_true_labels = self.get_true_labels(tmp_df)
            if phase != 'val':
                df_dict[phase] = {
                    'image_path': tmp_img_list, 'labels': tmp_true_labels
                }
            else:
                val_img_list0, val_img_list1, val_labels0, val_labels1 = train_test_split(
                    tmp_img_list, tmp_true_labels, test_size=0.5, random_state=42)
                df_dict['val0'] = {
                    'image_path': val_img_list0, 'labels': val_labels0
                }
                df_dict['val1'] = {
                    'image_path': val_img_list1, 'labels': val_labels1
                }
                
        #print(df_dict['train'])
        return df_dict

    def get_true_labels(self, df):
        true_labels = np.array(df[self.cat2idx.keys()].values)
        true_labels = np.nan_to_num(true_labels, nan=-1).astype(np.int32)
        return true_labels
       
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item, target = self.img_list[index], self.labels[index]
        return self.get(item), target

    def get_number_classes(self):
        return self.num_classes


def generate_noisy_labels(labels, noise_type, noise_rate, nb_classes, random_seed):
  #  labels = true_labels.copy()
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

