import json
import hashlib
import numpy as np 
import torch 
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime  
from pprint import pprint
from torch.utils.data import DataLoader 
from transformers import HfArgumentParser 

from .train_utils import get_encoder, get_pretrained_model 
from .trainer import VAETrainer
from argument import (
    CustomTrainingArguments,
    DataTrainingArguments,
    ModelArguments
)
from dnn.utils import get_device 


@dataclass
class VAETrainingArguments(CustomTrainingArguments):
    pretrained_clf: str = field(
        default='',
        metadata={'help': 'the path to load the pretrained classifier'}
    )
    beta: float = field(
        default=0.0001,
        metadata={'help': 'beta in beta-vae'}
    )
    post_model: str = field(
        default='lsnpc',
        metadata={'help': 'the post method name'}
    )
    grad_norm: int = field(
        default=2,
        metadata ={'help': 'gradient clipping norm'}
    )
    semi_sup: bool = field(
        default=False,
        metadata={'help': 'if to run semi supervised learning'}
    )
    is_ablation: bool = field(
        default=False,
        metadata={'help': 'if to run ablation study'}
    )

@dataclass
class VAEModelArguments(ModelArguments):
    nu0: float = field(
        default=2,
        metadata={'help': 'nu0 for the student-T degree of freedom'}
    )
    nu: float = field(
        default=2,
        metadata={'help': 'nu for the student-T degree of freedom'}
    )
    latent_dim: int = field(
        default=64,
        metadata={'help': 'latent size for MLP'}
    )
    label_emb_dim: int = field(
        default=128,
        metadata={'help': 'label embedding size'}
    )
    eta: float = field(
        default=0.5,
        metadata={'help': 'eta weight in the proposal for semi-sup'}
    )
 
 
parser = HfArgumentParser((
    VAEModelArguments, 
    DataTrainingArguments, 
    VAETrainingArguments
))
model_args, data_args, train_args = parser.parse_args_into_dataclasses()
  
np.random.seed(train_args.seed)
torch.manual_seed(train_args.seed)
torch.cuda.manual_seed(train_args.seed)

if train_args.is_ablation:
    from nlc.nlc_vae_gauss import (
        CorrectionLoss,
        MlcEncoderY,
        MlcEncoderZ,
        MlcDecoderY,
        MlcDecoderZ,
        NoisyLabelCorrectionVAE
    )
else:
    from nlc.nlc_vae import (
        CorrectionLoss,
        MlcEncoderY,
        MlcEncoderZ,
        MlcDecoderY,
        MlcDecoderZ,
        NoisyLabelCorrectionVAE
    )

if data_args.dataset.lower() in ['nuswide', 'coco']:
    from data_process import data_utils_old as data_utils 
    print('Importing the old data process for COCO/NUSWIDE')
else:
    from data_process import data_utils 

def train_lsnpc():
    if train_args.semi_sup:
        train_args.post_model += '_semi'

    if train_args.is_ablation:
        train_args.post_model += '_gauss'

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

        print(test_dataset)
 
    encoder, emb_size = get_encoder(model_args.img_encoder)
 
    pretrained_clf = get_pretrained_model(
        model_args.clf_name, train_args.pretrained_clf, 
        encoder, emb_size, n_labels
    )
 
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_args.batch_size,
        num_workers=data_args.num_workers,
        drop_last=True,
        shuffle=True,
        pin_memory=True
    )
    val_loader0 = DataLoader(
        dataset=val_dataset0,
        batch_size=train_args.batch_size*2,
        num_workers=data_args.num_workers,
        drop_last=True,
        shuffle=False,
        pin_memory=True
    )
    if train_args.semi_sup:
        clean_set_loader = DataLoader(
            dataset=val_dataset1,
            batch_size=train_args.batch_size,
            num_workers=data_args.num_workers,
            drop_last=True,
            shuffle=True,
            pin_memory=True
        )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=train_args.batch_size*4,
        num_workers=data_args.num_workers,
        drop_last=False,
        shuffle=False,
        pin_memory=True
    )

    # every dataset except COCO
    # latent_dim = 64 
    # label_emb_dim = 128

    # COCO
    latent_dim = model_args.latent_dim
    label_emb_dim = model_args.label_emb_dim 
    
    dp = 0.7 
    for run_index in range(train_args.n_repeats):
        arg_dict['run_index'] = run_index   
        
        # Create UID for saving relevant files (model, configuration, and summary)
        uid = hashlib.md5(json.dumps(arg_dict, sort_keys=True).encode('utf-8')).hexdigest() 

        arg_dict['uid'] = uid
        # Time added after the uid is created
        arg_dict['time'] = datetime.now().strftime("%Y%m%d_%H_%M_%S")
     
        data_enc = deepcopy(pretrained_clf.encoder)
        # data_enc = deepcopy(encoder)  
        # data_enc, emb_size = get_encoder('vit224')
        for p in data_enc.parameters():
            p.requires_grad = True

        encoder_y = MlcEncoderY(
            data_enc,  
            latent_dim, 
            n_labels, 
            emb_size,
            label_emb_dim,
            model_args.nu0,
            dp=dp
        )

        encoder_z = MlcEncoderZ(latent_dim, latent_dim, dp=dp)

        decoder_y = MlcDecoderY(
            data_enc,  
            latent_dim, 
            n_labels,
            emb_size, 
            label_emb_dim,
            dp=dp
        ) 

        decoder_z = MlcDecoderZ(latent_dim, latent_dim, dp=dp)
      
        model = NoisyLabelCorrectionVAE(
            encoder_y, 
            encoder_z,
            decoder_y, 
            decoder_z, 
            pretrained_clf,
            nu=model_args.nu,
            eta=model_args.eta 
        ) 

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_args.lr, 
            weight_decay=train_args.weight_decay
        ) 

        # Loss for multi-label classification
        loss_fn = CorrectionLoss(beta=train_args.beta)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_args.n_train_epoch, #eta_min=train_args.lr/1000
        )

        trainer = VAETrainer(
            model=model,
            n_labels=n_labels,
            loss_fn=loss_fn,
            optimizer=optimizer,
            arg_dict=arg_dict,
            lr_scheduler=lr_scheduler,
            device=get_device(), 
            train_on_val=train_args.semi_sup, 
            grad_norm=train_args.grad_norm,
            eval_test_at_final_loop_only=train_args.eval_test_at_final_loop_only, 
            metric_storing_path=f"./results/{arg_dict['dataset']}_results.csv"
        )

        trainer.train_model(
            train_args.n_train_epoch, 
            train_loader,
            val_loader0,
            test_loader,
            verbose=True,
            clean_set_loader=(None if not train_args.semi_sup else clean_set_loader)
        )

        print(f'Round {run_index} finished. \n\n')


if __name__ == '__main__':
    train_lsnpc()
