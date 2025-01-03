import json
import hashlib
import numpy as np
import torch
import torch.nn as nn
import tqdm
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from packaging import version
from pathlib import Path
from pprint import pprint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50, ResNet50_Weights
from transformers import HfArgumentParser, TrainingArguments
from transformers import AutoImageProcessor, ViTModel

import dnn.hlc as hlc
from .trainer import NPCModTrainer
from argument import (
    CustomTrainingArguments,
    DataTrainingArguments,
    ModelArguments
)
from dnn.mlc import MultilabelClassifier, ViTModelWrapper
from dnn.utils import get_device
from metrics import test
from nlc.npc_mod import ( 
    CorrectionLoss,
    EncoderCopulasWrapper,
    MLCEncoder, 
    MLCDecoder, 
    NoisyLabelCorrectionVAE
)


@dataclass
class VAETrainingArguments(CustomTrainingArguments):
    pretrained_clf: str = field(
        default=None,
        metadata={'help': 'the path to load the pretrained classifier'}
    )
    beta: float = field(
        default=0.0001,
        metadata={'help': 'beta in beta-vae'}
    ) 
    use_copula: bool = field(
        default=True,
        metadata={'help': 'if we should use copula for sampling multi-Bernoulli'}
    )
    grad_norm: int = field(
        default=2,
        metadata ={'help': 'gradient clipping norm'}
    )
    post_model: str = field(
        default='npc mod',
        metadata={'help': 'the post method name'}
    )
    semi_sup: bool = field(
        default=False,
        metadata={'help': 'if to run semi supervised learning'}
    )

 
parser = HfArgumentParser((
    ModelArguments, 
    DataTrainingArguments, 
    VAETrainingArguments
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


def train_mlnlc(run_index=0):
    arg_dict = {
        **vars(model_args), **vars(data_args), **vars(train_args)
    }
    pprint(arg_dict)
    # Create UID for saving relevant files (model, configuration, and summary)
    uid = hashlib.md5(json.dumps(arg_dict, sort_keys=True).encode('utf-8')).hexdigest()

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
        # Using 0.1.0
        # encoder = resnet50(pretrained=True)
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
    elif model_args.clf_name == 'addgcn':
        pretrained_clf = hlc.get_model(encoder, emb_size, n_labels)
    elif model_args.clf_name == 'hlc':
        pretrained_clf = hlc.get_model(encoder, emb_size, n_labels)
    else:
        raise AttributeError('Not recognized classifier')

    pretrained_clf.load_state_dict(
        torch.load(train_args.pretrained_clf, weights_only=True)
    ) 
    for p in pretrained_clf.parameters():
        p.requires_grad = False
 
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_args.batch_size,
        num_workers=data_args.num_workers,
        drop_last=True,
        shuffle=True,
        pin_memory=True)
    val_loader = DataLoader(
        dataset=val_dataset0,
        batch_size=train_args.batch_size,
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
   
    latent_dim = 64 
    label_emb_dim = 128

    for run_index in range(train_args.n_repeats):
        arg_dict['run_index'] = run_index   
        # Create UID for saving relevant files (model, configuration, and summary)
        uid = hashlib.md5(json.dumps(arg_dict, sort_keys=True).encode('utf-8')).hexdigest() 

        arg_dict['uid'] = uid
        # Time added after the uid is created
        arg_dict['time'] = datetime.now().strftime("%Y%m%d_%H_%M_%S")
 
        data_encoder = deepcopy(pretrained_clf.encoder)
        for p in data_encoder.parameters():
            p.requires_grad = True

        if train_args.use_copula:
            encoder = EncoderCopulasWrapper(
                MLCEncoder(
                    data_encoder,  
                    latent_dim, 
                    n_labels, 
                    emb_size,
                    label_emb_dim
                ),
                label_emb_dim
            )

            decoder = EncoderCopulasWrapper(
                MLCDecoder(
                    data_encoder,  
                    latent_dim, 
                    n_labels, 
                    emb_size,
                    label_emb_dim
                ),
                label_emb_dim
            )
        else:
            encoder = MLCEncoder(
                data_encoder,  
                latent_dim, 
                n_labels, 
                emb_size,
                label_emb_dim
            )

            decoder = MLCDecoder(
                data_encoder,  
                latent_dim, 
                n_labels,
                emb_size, 
                label_emb_dim
            ) 
        
        model = NoisyLabelCorrectionVAE(
            encoder, 
            decoder, 
            pretrained_clf, 
            use_copula=train_args.use_copula
        ) 

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_args.lr, 
            weight_decay=train_args.weight_decay
        )

        # Loss for multi-label classification
        loss_fn = CorrectionLoss(beta=train_args.beta)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=1e-8
        )

        trainer = NPCModTrainer(
            model=model,
            n_labels=n_labels,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            arg_dict=arg_dict,
            device=get_device(),  
            train_on_val=train_args.semi_sup,
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

        res_path = Path(train_args.result_dir) / (
            f'./mlnlc/{data_args.dataset}_{data_args.noise_type}_'
            f'{data_args.noise_rate}_{model_args.img_encoder}_'
            f'ep{train_args.n_train_epoch}_rd{run_index}/'
        )
        trainer.save_model(arg_dict, res_path)


if __name__ == '__main__':
    train_mlnlc()

