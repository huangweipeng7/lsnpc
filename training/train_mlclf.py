"""Training script for standard Multi-Label Classifier (MLCLF).

This script uses the refactored base training pipeline to reduce code duplication.
"""

from dataclasses import dataclass, field
 
import torch.nn as nn

from argument import DataTrainingArguments, ModelArguments, CustomTrainingArguments
from .base_pipeline import StandardTrainingPipeline, generate_run_uid, create_timestamp
from .train_utils import get_encoder, load_dataset_module
from dnn.mlc import MultilabelClassifier, ViTModelWrapper


@dataclass
class MLCLFTrainingArguments(CustomTrainingArguments):
    """Training arguments for standard classifier training."""
    
    loss_fn: str = field(
        default='bce',
        metadata={'help': 'Loss function type (bce or asl)'}
    )


class MLCLFPipeline(StandardTrainingPipeline):
    """Training pipeline for standard multi-label classifier."""
    
    def __init__(self, model_args, data_args, train_args, n_labels=None):
        super().__init__(model_args, data_args, train_args, n_labels=n_labels)
        
    def build_model(
        self,
        encoder: nn.Module,
        emb_size: int,
        n_labels: int,
    ) -> nn.Module:
        """Build standard multi-label classifier.
        
        Args:
            encoder: Encoder architecture.
            emb_size: Embedding size.
            n_labels: Number of labels.
            
        Returns:
            Multi-label classifier model.
        """
        return MultilabelClassifier(encoder, emb_size, n_labels)
    
    def get_loss_fn(self) -> nn.Module:
        """Get loss function based on configuration.
        
        Returns:
            Loss function module.
        """
        if getattr(self.model_args, 'loss_fn', 'bce') == 'asl':
            from dnn.losses import AsymmetricLoss
            return AsymmetricLoss()
        else:
            return nn.BCEWithLogitsLoss()
    
    def train(self):
        """Execute MLCLF training pipeline."""
        # Setup logging
        self.setup_logging()
        
        # Setup seeds
        self.setup_seeds()
        
        # Load dataset module
        data_utils = self.load_dataset_module()
        
        # Print arguments
        from pprint import pprint
        arg_dict = {
            **self.model_args.__dict__,
            **self.data_args.__dict__,
            **self.train_args.__dict__,
        }
        pprint(arg_dict)
        
        # Load data
        data = data_utils.load_data(self.data_args)
        n_labels = data['n_labels']
        
        # Cache n_labels to avoid reloading data in _get_n_labels()
        self._n_labels = n_labels
        
        # Verify data consistency if requested
        if getattr(self.train_args, 'checksum', False):
            from .train_utils import verify_data_consistency
            verify_data_consistency(data)
        
        # Load encoder
        encoder, emb_size = get_encoder(self.model_args.img_encoder)
        
        # Create data loaders (no clean loader needed for standard training)
        loaders = self.create_data_loaders(data, include_clean_loader=False)
        
        # Training loop
        for run_index in range(self.train_args.n_repeats):
            # Generate run UID
            uid = generate_run_uid(arg_dict, run_index)
            run_arg_dict = arg_dict.copy()
            run_arg_dict['uid'] = uid
            run_arg_dict['time'] = create_timestamp()
            run_arg_dict['run_index'] = run_index
            
            # Build model
            model = self.build_model(encoder, emb_size, n_labels)
            
            # Get optimizer and scheduler
            optimizer = self.get_optimizer(model)
            loss_fn = self.get_loss_fn()
            lr_scheduler = self.get_lr_scheduler(optimizer)
            
            # Run training loop
            self.run_training_loop(
                run_index=run_index,
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                lr_scheduler=lr_scheduler,
                loaders=loaders,
                extra_args=run_arg_dict,
            )


def parse_arguments():
    """Parse command-line arguments using HuggingFace parser."""
    from transformers import HfArgumentParser
    
    parser = HfArgumentParser((
        ModelArguments,
        DataTrainingArguments,
        CustomTrainingArguments,
    ))
    return parser.parse_args_into_dataclasses()


def train_mlclf():
    """Main entry point for MLCLF training."""
    # Parse arguments
    model_args, data_args, train_args = parse_arguments()
    
    # Create and run pipeline
    pipeline = MLCLFPipeline(model_args, data_args, train_args)
    pipeline.train()


if __name__ == '__main__':
    train_mlclf()
