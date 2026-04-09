import numpy as np
import orjson
import torch
import torch.nn as nn
from datetime import datetime 
from pathlib import Path
from pprint import pprint
from transformers import HfArgumentParser
from transformers.optimization import GrokOptimizer

from trainers import NPCModTrainer
from .train_utils import get_encoder, get_pretrained_model
from argument import (
    CustomTrainingArguments,
    DataTrainingArguments,
    ModelArguments
) 
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
        default='',
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
        default='npc_mod',
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


from data_process import data_utils 


def train_mlnlc(run_index=0):
    arg_dict = {
        **vars(model_args), **vars(data_args), **vars(train_args)
    }
    pprint(arg_dict)
    # Create UID for saving relevant files (model, configuration, and summary)
    uid = hashlib.md5(orjson.dumps(arg_dict, option=orjson.OPT_SORT_KEYS)).hexdigest()

    data = data_utils.load_data(data_args)
    train_dataset, val_dataset, clean_val_dataset, test_dataset, n_labels = \
        data['train_dataset'], data['val_dataset'], data['clean_val_dataset'], data['test_dataset'], data['n_labels']

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
        pin_memory=True)
    val_loader = DataLoader(
        dataset=val_dataset,
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
        uid = hashlib.md5(orjson.dumps(arg_dict, option=orjson.OPT_SORT_KEYS)).hexdigest() 

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

        optimizer = GrokOptimizer(
            model.parameters(),
            lr=train_args.lr, 
            weight_decay=train_args.weight_decay,
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
            train_on_val=train_args.semi_sup,
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

        res_path = Path(train_args.result_dir) / (
            f'./mlnlc/{data_args.dataset}_{data_args.noise_type}_'
            f'{data_args.noise_rate}_{model_args.img_encoder}_'
            f'ep{train_args.n_train_epoch}_rd{run_index}/'
        )
        trainer.save_model(arg_dict, res_path)


if __name__ == '__main__':
    train_mlnlc()

