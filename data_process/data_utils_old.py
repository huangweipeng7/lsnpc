import argparse
import logging
import numpy as np
import sys
import torch
import torchvision.transforms as transforms

from .coco_old import COCO2014
from utils import MultiScaleCrop, Warp 

logger = logging.getLogger(__name__)


def load_data(args):
    print('test', args.root)
    seed = 1000
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    train_transform = transforms.Compose([
            MultiScaleCrop(args.image_size, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    val_transform= transforms.Compose([
            Warp(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    if args.dataset=='coco':
        train_dataset = COCO2014(args.root,
                                 phase='train', 
                                 noise_type=args.noise_type, 
                                 noise_rate=args.noise_rate, 
                                 transform=train_transform,
                                 split_per=args.split_percentage,
                                 random_seed=seed)

        val_dataset0 = COCO2014(args.root,
                                 phase='val0', 
                                 noise_type=args.noise_type, 
                                 noise_rate=args.noise_rate, 
                                 transform=val_transform,
                                 split_per=args.split_percentage,
                                 random_seed=seed)

        val_dataset1 = COCO2014(args.root,
                                 phase='val1', 
                                 noise_type=args.noise_type, 
                                 noise_rate=args.noise_rate, 
                                 transform=train_transform,
                                 split_per=args.split_percentage,
                                 random_seed=seed) 

        test_dataset = COCO2014(args.root, 
                                phase='test',
                                transform=val_transform)
    else:
        raise Exception('Only COCO is supported as this manner is a rather runtime inefficient but memory efficient.')

    print('Done!')
    n_labels = 80

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