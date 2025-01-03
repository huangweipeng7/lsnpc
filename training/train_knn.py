import json
import hashlib
import numpy as np
import torch
import torch.nn as nn
import tqdm
from copy import deepcopy
from dataclasses import dataclass, field
from packaging import version
from pathlib import Path
from pprint import pprint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50, ResNet50_Weights
from transformers import HfArgumentParser, TrainingArguments
from transformers import AutoImageProcessor, LevitModel, ViTModel
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier

import dnn.hlc as hlc
from .trainer import KNNTrainer
from argument import (
    CustomTrainingArguments,
    DataTrainingArguments,
    ModelArguments
)
from data_process import data_utils as data_utils
from dnn.mlc import MultilabelClassifier, ViTModelWrapper
from dnn.utils import freeze_param, get_device
from metrics import test 


@dataclass
class KNeighborsTrainingArguments(CustomTrainingArguments):
    pretrained_clf: str = field(
        default=None,
        metadata={'help': 'the path to load the pretrained classifier'}
    )
    k: float = field(
        default=10,
        metadata={'help': 'the number of neighbors'}
    )
    post_model: str = field(
        default='knn',
        metadata={'help': 'the post method name'}
    )

parser = HfArgumentParser((
    ModelArguments, 
    DataTrainingArguments, 
    KNeighborsTrainingArguments
))
model_args, data_args, train_args = parser.parse_args_into_dataclasses()
   
np.random.seed(train_args.seed)
torch.manual_seed(train_args.seed)
torch.cuda.manual_seed(train_args.seed)
 

def train_knn():
    arg_dict = {
        **vars(model_args), **vars(data_args), **vars(train_args)
    }
 
    data = data_utils.load_data(data_args)
    train_dataset, val_dataset0, val_dataset1, test_dataset, n_labels = \
        data['train_dataset'], data['val_dataset0'], data['val_dataset1'], data['test_dataset'], data['n_labels']
 
    # Check consistency
    if train_args.checksum:
        print('val_dataset true labels', (val_dataset.true_labels[:10]))
        print('val_dataset labels', val_dataset.labels[:10])
        print('test_dataset true labels', (test_dataset.true_labels[:10]))

    print(test_dataset)
  
    if model_args.img_encoder == 'resnet50': 
        encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
        encoder = torch.nn.Sequential(*(list(encoder.children())[:-1]))
        encoder.fc = nn.Flatten()
        emb_size = 2048 
    elif model_args.img_encoder == 'levit':
        encoder = ViTModelWrapper(
            LevitModel.from_pretrained(
                'local_models/levit', local_files_only=True
            )
        )
        emb_size = 384
    else:
        raise AttributeError('Image feature encoder is not defined...')
    
    if model_args.clf_name == 'mlclf': 
        pretrained_clf = MultilabelClassifier(encoder, emb_size, n_labels)
    elif model_args.clf_name in ['addgcn', 'hlc']:
        pretrained_clf = hlc.get_model(encoder, emb_size, n_labels)
    else:
        raise AttributeError('Not recognized classifier')

    for run_index in range(train_args.n_repeats):
        arg_dict['run_index'] = run_index
        # Create UID for saving relevant files (model, configuration, and summary)
        uid = hashlib.md5(json.dumps(arg_dict, sort_keys=True).encode('utf-8')).hexdigest() 

        arg_dict['uid'] = uid
        # Time added after the uid is created
        arg_dict['time'] = datetime.now().strftime("%Y%m%d_%H_%M_%S")

        pretrained_clf.load_state_dict(
            torch.load(train_args.pretrained_clf, weights_only=True)
        )
        pretrained_clf.apply(freeze_param)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=train_args.batch_size,
            num_workers=data_args.num_workers,
            drop_last=False,
            shuffle=False,
            pin_memory=True
        )
        val_loader = DataLoader(
            dataset=val_dataset0,
            batch_size=train_args.batch_size,
            num_workers=data_args.num_workers,
            drop_last=False,
            shuffle=False,
            pin_memory=True
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=train_args.batch_size,
            num_workers=data_args.num_workers,
            drop_last=False,
            shuffle=False,
            pin_memory=True
        )
 
        trainer = KNNTrainer(
            pretrained_clf = pretrained_clf,
            model = KNeighborsClassifier(n_neighbors=train_args.k, weights='distance'),
            n_labels = n_labels,
            arg_dict = arg_dict,
            encoder = pretrained_clf.encoder if hasattr(pretrained_clf, 'encoder') else encoder,
        )

        trainer.train_model(
            train_loader, 
            val_loader, 
            test_loader,
            verbose=True
        )


if __name__ == '__main__':
    train_knn()

