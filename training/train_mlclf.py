import json
import hashlib
import numpy as np
import torch
import torch.nn as nn 
from datetime import datetime 
from pprint import pprint
from torch.utils.data import DataLoader  
from transformers import HfArgumentParser 
 

from .trainer import Trainer
from .train_utils import get_encoder 
from argument import (
    CustomTrainingArguments,
    DataTrainingArguments,
    ModelArguments
)
from dnn.mlc import MultilabelClassifier, ViTModelWrapper
from dnn.utils import freeze_param, get_device
from metrics import test 
from packaging import version


parser = HfArgumentParser((
    ModelArguments, 
    DataTrainingArguments, 
    CustomTrainingArguments
))
model_args, data_args, train_args = parser.parse_args_into_dataclasses()

np.random.seed(train_args.seed)
torch.manual_seed(train_args.seed)
torch.cuda.manual_seed(train_args.seed)

if data_args.dataset.lower() in ['nuswide', 'coco']:
    from data_process import data_utils_old as data_utils 
    print('Importing the old data process for COCO/NUSWIDE')
else:
    from data_process import data_utils 


def train_clf():
    arg_dict = {
        **vars(model_args), **vars(data_args), **vars(train_args)
    } 
    pprint(arg_dict)

    data = data_utils.load_data(data_args)
    train_dataset, val_dataset0, _, test_dataset, n_labels = \
        data['train_dataset'], data['val_dataset0'], data['val_dataset1'], data['test_dataset'], data['n_labels']

    # print(f'Training examples: {len(train_dataset.labels)}')
    # print(f'Val0 examples: {len(val_dataset0.labels)}')
    # print(f'Val1 examples: {len(val_dataset1.labels)}')
    # print(f'Test examples: {len(test_dataset.labels)}')
    # print(f'Number of labels: {n_labels}')

    # Check consistency
    if train_args.checksum:
        print('val_dataset true labels', (val_dataset.true_labels[:10]))
        print('val_dataset labels', val_dataset.labels[:10])
        print('test_dataset true labels', (test_dataset.true_labels[:10]))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_args.batch_size,
        num_workers=data_args.num_workers,
        drop_last=False,
        shuffle=True,
        pin_memory=True)
    val_loader = DataLoader(
        dataset=val_dataset0,
        batch_size=train_args.batch_size*2,
        num_workers=data_args.num_workers,
        drop_last=False,
        shuffle=False,
        pin_memory=True)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=train_args.batch_size*2,
        num_workers=data_args.num_workers,
        drop_last=False,
        shuffle=False,
        pin_memory=True)
  
    for run_index in range(train_args.n_repeats):
        arg_dict['run_index'] = run_index 
        # Create UID for saving relevant files (model, configuration, and summary)
        uid = hashlib.md5(json.dumps(arg_dict, sort_keys=True).encode('utf-8')).hexdigest() 

        arg_dict['uid'] = uid
        # Time added after the uid is created
        arg_dict['time'] = datetime.now().strftime("%Y%m%d_%H_%M_%S")

        encoder, emb_size = get_encoder(model_args.img_encoder)

        model = MultilabelClassifier(encoder, emb_size, n_labels)

        # summary(model, (3,224,224), device='cpu')

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_args.lr,
            weight_decay=train_args.weight_decay
        )
        lr_scheduler = None
  
        # Loss for multi-label classification
        if model_args.loss_fn == 'asl':
            from dnn.losses import AsymmetricLoss
            loss_fn = AsymmetricLoss()
        else:
            loss_fn = nn.BCEWithLogitsLoss()

        trainer = Trainer(
            model=model,
            n_labels=n_labels,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=get_device(),
            arg_dict=arg_dict,
            eval_test_at_final_loop_only=train_args.eval_test_at_final_loop_only, 
            metric_storing_path=f"./results/{arg_dict['dataset']}_results.csv"
        )

        trainer.train_model(
            train_args.n_train_epoch, 
            train_loader,
            val_loader,
            test_loader,
            verbose=True
        )

        print(f'Round {run_index} finished. \n\n') 


if __name__ == '__main__':
    train_clf()

