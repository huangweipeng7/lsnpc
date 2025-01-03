import json
import hashlib
import numpy as np
import torch
import torch.nn as nn
import tqdm
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50, ResNet50_Weights
from transformers import (
    HfArgumentParser, 
    TrainingArguments,
    ViTModel
)
from torchsummary import summary
from datetime import datetime

import dnn.hlc as hlc
from .trainer import HLCTrainer
from argument import (
    CustomTrainingArguments,
    DataTrainingArguments,
    ModelArguments,
) 
from dnn.mlc import MultilabelClassifier, ViTModelWrapper
from dnn.utils import get_device
from metrics import test
from packaging import version


@dataclass
class HLCTrainingArguments(CustomTrainingArguments):
    beta: float = field(
        default=.5,
        metadata={'help': 'hyper-parameter for balancing instance-label dependence and label dependence in hlc'}
    )
    delta: float = field(
        default=.4,
        metadata={'help': 'hyper-parameter of hlc'}
    )
    epoch_update_start: int = field(
        default=5,
        metadata={'help': 'when to start correction for hlc'}
    )


parser = HfArgumentParser((
    ModelArguments, 
    DataTrainingArguments, 
    HLCTrainingArguments
))
model_args, data_args, train_args = parser.parse_args_into_dataclasses()
   
np.random.seed(train_args.seed)
torch.manual_seed(train_args.seed)
torch.cuda.manual_seed(train_args.seed)


if data_args.dataset.lower() == 'coco':
    from data_process import data_utils_old as data_utils 
    print('Importing the old data process for COCO')
else:
    from data_process import data_utils 


def train_hlc(run_index=0):
    arg_dict = {
        **vars(model_args), **vars(data_args), **vars(train_args)
    } 
    pprint(arg_dict)

    data = data_utils.load_data(data_args)
    train_dataset, val_dataset0, val_dataset1, test_dataset, n_labels = \
        data['train_dataset'], data['val_dataset0'], data['val_dataset1'], data['test_dataset'], data['n_labels'] 

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
        shuffle=False,
        pin_memory=True)
    val_loader = DataLoader(
        dataset=val_dataset0,
        batch_size=train_args.batch_size*4,
        num_workers=data_args.num_workers,  
        drop_last=False,
        shuffle=False,
        pin_memory=True)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=train_args.batch_size*4,
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

        if model_args.img_encoder == 'resnet50': 
            encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
            encoder = torch.nn.Sequential(*(list(encoder.children())[:-1]))
            encoder.fc = nn.Flatten()
            emb_size = 2048
        else:
            raise Exception('Image feature encoder is not defined...')

        if model_args.clf_name == 'mlclf': 
            model = MultilabelClassifier(encoder, emb_size, n_labels)
        elif model_args.clf_name == 'addgcn':
            model = hlc.get_model(encoder, emb_size, n_labels)
        elif model_args.clf_name == 'hlc':
            model = hlc.get_model(encoder, emb_size, n_labels)
        else:
            raise Exception('Not recognized classifier')
 
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_args.lr,
            weight_decay=train_args.weight_decay
        ) 

        # Loss for multi-label classification
        loss_fn = nn.BCEWithLogitsLoss()
        lr_scheduler = None
  
        trainer = HLCTrainer(
            model=model,
            n_labels=n_labels,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=get_device(),
            delta=train_args.delta,
            epoch_update_start=train_args.epoch_update_start,
            beta=train_args.beta,
            arg_dict=arg_dict,
            eval_test_at_final_loop_only=train_args.eval_test_at_final_loop_only,
            metric_storing_path=train_args.metric_storing_path
        )

        trainer.train_model(
            train_args.n_train_epoch, 
            train_loader,
            val_loader,
            test_loader,
            verbose=True
        )

        print(f'Round {run_index} finished. \n\n')

    #res_path = Path(train_args.result_dir) / (
    #    f'./hlc/{data_args.dataset}_{data_args.noise_type}_'
    #    f'{data_args.noise_rate}_{model_args.img_encoder}_'
    #    f'ep{train_args.n_train_epoch}_rd{run_index}/'
    #)
    #trainer.save_model(arg_dict, res_path)

    # return trainer 


if __name__ == '__main__':
    train_hlc()

