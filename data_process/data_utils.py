import argparse
import logging
import numpy as np
import sys
import torch
import torchvision.transforms as transforms
from functools import partial
from PIL import Image

from .coco import COCO2014
from .tomato import Tomato
from .voc import Voc2007, Voc2012
from utils import MultiScaleCrop, Warp


logger = logging.getLogger(__name__)


def batch_transform(batch, transform):
    batch['data'] = [
        transform(x) for x in iter(batch['data'])
    ]
    batch['labels'] = [
        torch.tensor(x) for x in iter(batch['labels'])
    ]
    return batch


def load_data(args):    
    train_transform = transforms.Compose([
        MultiScaleCrop(
            args.image_size, 
            scales=(1.0, 0.875, 0.75, 0.66, 0.5), 
            max_distort=2
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])]
        )
    
    val_transform = transforms.Compose([
        Warp(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    if args.dataset=='coco':
        DataClass = COCO2014 
    elif args.dataset=='voc2007':
        DataClass = Voc2007
    elif args.dataset=='voc2012':
        DataClass = Voc2012
    elif args.dataset=='tomato':
        DataClass = Tomato
    else:
        raise Exception(f"Dataset not prepared: {args.dataset}")
   
    data_class = DataClass(
        args.root, 
        noise_type=args.noise_type, 
        noise_rate=args.noise_rate,  
        split_per=args.split_percentage,
        num_workers=args.num_workers,
        noisy_val=args.noisy_val
    )

    train_dataset = data_class.train_data.with_format(
        'torch', columns=['labels'], output_all_columns=True
    )
    
    train_dataset.set_transform(
        partial(batch_transform, transform=train_transform)
    )  
    val_dataset0 = data_class.val_data0.with_transform(
        partial(batch_transform, transform=val_transform)
    ) 
    val_dataset1 = data_class.val_data1.with_transform(
        partial(batch_transform, transform=train_transform)
    ) 
    test_dataset = data_class.test_data.with_transform(
        partial(batch_transform, transform=val_transform)
    ) 

    print('Done!')
    n_labels = data_class.get_number_classes()

    return {
        'train_dataset': train_dataset, 
        'val_dataset0': val_dataset0,  
        'val_dataset1': val_dataset1, 
        'test_dataset': test_dataset,
        'n_labels': n_labels
    }


if __name__ == '__main__':
    logging.basicConfig(
        #filename='myapp.log', 
        stream=sys.stdout, 
        level=logging.INFO
    )

    parser = argparse.ArgumentParser('Datautils')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--root', type=str, default='data/COCO/')
    parser.add_argument('--image_size', type=int, help='image size', default=224)
    parser.add_argument('--split_percentage', type=float, help='train and validation', default=0.9)
    parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric]', default='symmetric')
    parser.add_argument('--noise_rate', type=float, help='overall corruption rae, should be less than 1', default=0.4)
    parser.add_argument('--dataset', type=str, help='voc2007/2012, coco, tomato', default='2007')
    args = parser.parse_args()

    logger.info('Loading data')
    train_dataset, val_dataset, test_dataset = load_data(args)
   
    print('train image list', train_dataset.images)
    #print('train cat2idx', train_dataset.cat2idx)
    print('train labels', train_dataset.labels)
    print('train true labels', train_dataset.true_labels)
    print('val true labels', val_dataset.true_labels[0,:])
    print('test true labels', test_dataset.true_labels[0,:])
    # for dataset in [train_dataset, val_dataset, test_dataset]:
    #     print(np.asarray(dataset.labels == dataset.true_labels).sum(), np.asarray(dataset.true_labels).size)
