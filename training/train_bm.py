import numpy as np
import orjson
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from pprint import pprint
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from transformers import (
    HfArgumentParser,
    LevitModel,
    ViTModel
)

import dnn.hlc as hlc
from trainers import BalanceMixTrainer
from argument import (
    CustomTrainingArguments,
    DataTrainingArguments,
    ModelArguments
)
from dnn.mlc import MultilabelClassifier
from dnn.utils import freeze_param


parser = HfArgumentParser((
    ModelArguments, 
    DataTrainingArguments, 
    CustomTrainingArguments
))
model_args, data_args, train_args = parser.parse_args_into_dataclasses()

np.random.seed(train_args.seed)
torch.manual_seed(train_args.seed)
torch.cuda.manual_seed(train_args.seed)

from data_process import data_utils 


def train_clf():
    arg_dict = {
        **vars(model_args), **vars(data_args), **vars(train_args)
    } 
    pprint(arg_dict)

    data = data_utils.load_data(data_args)
    train_dataset, val_dataset, _, test_dataset, n_labels = \
        data['train_dataset'], data['val_dataset'], data['clean_val_dataset'], data['test_dataset'], data['n_labels']

    # print(f'Training examples: {len(train_dataset.labels)}')
    # print(f'Val examples: {len(val_dataset.labels)}')
    # print(f'Clean val examples: {len(clean_val_dataset.labels)}')
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
        dataset=val_dataset,
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
        uid = hashlib.md5(orjson.dumps(arg_dict, option=orjson.OPT_SORT_KEYS)).hexdigest() 

        arg_dict['uid'] = uid
        # Time added after the uid is created
        arg_dict['time'] = datetime.now().strftime("%Y%m%d_%H_%M_%S")

        if model_args.img_encoder == 'resnet50':
            # Using 0.1.0
            #encoder = resnet50(pretrained=True)
            encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
            encoder = torch.nn.Sequential(*(list(encoder.children())[:-1]))
            encoder.fc = nn.Flatten()
            emb_size = 2048
        elif model_args.img_encoder == 'vit224':
            encoder = ViTModelWrapper(
                ViTModel.from_pretrained(
                    'local_models/levit', local_files_only=True
                )
            )
            emb_size = 768 
        elif model_args.img_encoder == 'levit':
            encoder = ViTModelWrapper(
                LevitModel.from_pretrained(
                    'local_models/levit', local_files_only=True
                )
            )
            emb_size = 384
        else:
            raise Exception('Image feature encoder is not defined...')

        if model_args.clf_name in ['mlclf', 'mlbm']: 
            model = MultilabelClassifier(encoder, emb_size, n_labels)
        elif model_args.clf_name == 'addgcn':
            model = hlc.get_model(n_labels)
        elif model_args.clf_name == 'hlc':
            model = hlc.get_model(n_labels)
        else:
            raise Exception('Not recognized classifier')
        #summary(model, (3,224,224), device='cpu')

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_args.lr,
            weight_decay=train_args.weight_decay
        )
        lr_scheduler = None
                # BalanceMix paper's optimizer settings
        # optimizer = torch.optim.SGD(
        #     model.parameters(),
        #     lr=train_args.lr,           # Initial learning rate as per paper
        #     momentum=0.9,      # Momentum as per paper
        #     weight_decay=1e-4  # Weight decay as per paper
        # )
        
        # # Cosine annealing scheduler without restart
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, 
        #     T_max=train_args.n_train_epoch,  # Total number of epochs
        #     eta_min=0  # Minimum learning rate
        # )
  
        # Loss for multi-label classification
        loss_fn = nn.BCEWithLogitsLoss()

        trainer = BalanceMixTrainer(
            model=model,
            n_labels=n_labels,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            arg_dict=arg_dict,
            eval_test_at_final_loop_only=train_args.eval_test_at_final_loop_only, 
            metric_storing_path=f"./results/{arg_dict['dataset']}_results.csv",
            accelerator=self.accelerator,
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

