from dataclasses import dataclass, field
from typing import Dict, List, Optional
from transformers import (
    HfArgumentParser, 
    TrainingArguments
)

 
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset: str = field(
        default='voc2007',
        metadata={'help': '[voc2007/2012, tomato, coco]'},
    )
    noise_rate: Optional[float] = field(
        default=0.4,
        metadata={'help': 'overall corruption rate, should be less than 1'},
    )
    noise_type: Optional[str] = field(
        default='symmetric',
        metadata={'help': '[pairflip, symmetric]'},
    ) 
    image_size: Optional[int] = field(
        default=224,
        metadata={'help': 'image size, defaulted to 224'},
    )
    split_percentage: Optional[float] = field(
        default=0.9,
        metadata={'help': 'train and validation'},
    )
    root: Optional[str] = field(
        default='./data/',
        metadata={'help': './data/'},
    )
    num_workers: int = field(
        default=2,
        metadata={'help': 'number of workers for loading data'}
    )   
    noisy_val: bool = field(
        default=False,
        metadata={'help': 'whether to set noisy validation labels'}
    ) 


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    clf_name: Optional[str] = field(
        default='mlclf',
        metadata={'help': '[mlclf, addgcn, hlc], multi-label classifier, will also be used as the name to be stored'}
    )
    img_encoder: Optional[str] = field(
        default='resnet50',
        metadata={'help': 'image encoder'}
    )
    cache_dir: Optional[str] = field(
        default=None, 
        metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    

@dataclass
class CustomTrainingArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    result_dir: Optional[str] = field(
        default='./output/standard/',
        metadata={'help': 'dir to save result files'},
    )
    lr: Optional[float] = field(
        default=5e-5,
        metadata={'help': 'learning rate'}
    )
    n_train_epoch: Optional[int] = field(
        default=30,
        metadata={'help': 'number of training epochs'}
    )
    weight_decay: Optional[float] = field(
        default=4e-5,
        metadata={'help': 'weight decay'}
    )
    batch_size: Optional[int] = field(
        default=128,
        metadata={'help': 'batch size'}
    )
    checksum: bool = field(
        default=False,
        metadata={'help': 'a flag for if to print additional info for debugging'}
    )
    seed: int = field(
        default=128,
        metadata={'help': 'random seed'}
    )
    n_repeats: int = field(
        default=1,
        metadata={'help': 'number of repeated experiments'}
    )
    patience: int = field(
        default=100,
        metadata={'help': 'patience for early stopping, only used for post processing'}
    )
    eval_test_at_final_loop_only: bool = field(
        default=False,
        metadata={'help': 'if to evaluate test data only at the final loop'}
    ) 
    metric_storing_path: Optional[str] = field(
        default='./runs/results.csv',
        metadata={'help': 'the path for storing the metric results'},
    )
