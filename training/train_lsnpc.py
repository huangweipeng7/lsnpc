"""Training script for LSNPC (Latent Shift Noisy Label Correction) model.

This script uses the refactored base training pipeline to reduce code duplication.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Tuple

import torch
import torch.nn as nn

from argument import CustomTrainingArguments, DataTrainingArguments, ModelArguments
from .base_pipeline import VAETrainingPipeline, generate_run_uid, create_timestamp
from .train_utils import get_encoder, get_pretrained_model, load_dataset_module


@dataclass
class LSNPCModelArguments(ModelArguments):
    """Model arguments for LSNPC VAE architecture.
    
    Extends base ModelArguments with VAE-specific parameters.
    """
    
    nu0: float = field(
        default=2,
        metadata={'help': 'Degrees of freedom for prior distribution (nu0)'}
    )
    nu: float = field(
        default=2,
        metadata={'help': 'Degrees of freedom for student-T distribution (nu)'}
    )
    latent_dim: int = field(
        default=64,
        metadata={'help': 'Latent space dimensionality'}
    )
    label_emb_dim: int = field(
        default=128,
        metadata={'help': 'Label embedding dimensionality'}
    )
    eta: float = field(
        default=0.5,
        metadata={'help': 'Eta weight for semi-supervised learning loss'}
    )


@dataclass
class LSNPCTrainingArguments(CustomTrainingArguments):
    """Training arguments for LSNPC model.
    
    Extends CustomTrainingArguments with LSNPC-specific parameters.
    """
    
    pretrained_clf: str = field(
        default='',
        metadata={'help': 'Path to load pretrained classifier'}
    )
    beta: float = field(
        default=0.0001,
        metadata={'help': 'Beta parameter in beta-VAE'}
    )
    post_model: str = field(
        default='lsnpc',
        metadata={'help': 'Post-processing method name'}
    )
    grad_norm: int = field(
        default=2,
        metadata={'help': 'Gradient clipping norm'}
    )
    semi_sup: bool = field(
        default=False,
        metadata={'help': 'Enable semi-supervised learning'}
    )
    is_ablation: bool = field(
        default=False,
        metadata={'help': 'Enable ablation study mode'}
    )


class LSNPCPipeline(VAETrainingPipeline):
    """Training pipeline for LSNPC model."""
    
    def __init__(self, model_args, data_args, train_args):
        super().__init__(model_args, data_args, train_args)
        self.model_components = None
        
    def import_model_components(self) -> Tuple:
        """Import model components based on ablation study flag.
        
        Returns:
            Tuple of model classes.
        """
        if self.train_args.is_ablation:
            from nlc.nlc_vae_gauss import (
                CorrectionLoss,
                MlcEncoderY,
                MlcEncoderZ,
                MlcDecoderY,
                MlcDecoderZ,
                NoisyLabelCorrectionVAE,
            )
        else:
            from nlc.nlc_vae import (
                CorrectionLoss,
                MlcEncoderY,
                MlcEncoderZ,
                MlcDecoderY,
                MlcDecoderZ,
                NoisyLabelCorrectionVAE,
            )
        
        return (CorrectionLoss, MlcEncoderY, MlcEncoderZ, MlcDecoderY, 
                MlcDecoderZ, NoisyLabelCorrectionVAE)
    
    def build_model(
        self,
        pretrained_clf: nn.Module,
        encoder: nn.Module,
        emb_size: int,
        n_labels: int,
    ) -> nn.Module:
        """Build LSNPC model with encoders and decoders.
        
        Args:
            pretrained_clf: Pretrained classifier.
            encoder: Encoder architecture.
            emb_size: Embedding size.
            n_labels: Number of labels.
            
        Returns:
            LSNPC model.
        """
        # Import model components
        (CorrectionLoss, MlcEncoderY, MlcEncoderZ, MlcDecoderY, 
         MlcDecoderZ, NoisyLabelCorrectionVAE) = self.import_model_components()
        
        latent_dim = self.model_args.latent_dim
        label_emb_dim = self.model_args.label_emb_dim
        dropout = 0.1
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f'Building model with latent_dim: {latent_dim}, label_emb_dim: {label_emb_dim}')
        
        # Build encoder Y
        data_enc = deepcopy(pretrained_clf.encoder)
        for p in data_enc.parameters():
            p.requires_grad = True
        
        encoder_y = MlcEncoderY(
            data_enc,
            latent_dim,
            n_labels,
            emb_size,
            label_emb_dim,
            self.model_args.nu0,
            dp=dropout,
        )
        
        # Build encoder Z
        encoder_z = MlcEncoderZ(latent_dim, latent_dim, dp=dropout)
        
        # Build decoder Y
        decoder_y = MlcDecoderY(
            data_enc,
            latent_dim,
            n_labels,
            emb_size,
            label_emb_dim,
            dp=dropout,
        )
        
        # Build decoder Z
        decoder_z = MlcDecoderZ(latent_dim, latent_dim, dp=dropout)
        
        # Build main model
        model = NoisyLabelCorrectionVAE(
            encoder_y,
            encoder_z,
            decoder_y,
            decoder_z,
            pretrained_clf,
            nu=self.model_args.nu,
            eta=self.model_args.eta,
        )
        
        return model
    
    def get_loss_fn(self):
        """Get correction loss function."""
        (CorrectionLoss, _, _, _, _, _) = self.model_components
        return CorrectionLoss(beta=self.train_args.beta)
    
    def create_trainer(self, model, optimizer, loss_fn, lr_scheduler, loaders, arg_dict):
        """Create VAE trainer for LSNPC."""
        from trainers import VAETrainer
        
        return VAETrainer(
            model=model,
            n_labels=self._get_n_labels(),
            loss_fn=loss_fn,
            optimizer=optimizer,
            arg_dict=arg_dict,
            lr_scheduler=lr_scheduler,
            train_on_val=self.train_args.semi_sup,
            grad_norm=self.train_args.grad_norm,
            eval_test_at_final_loop_only=getattr(
                self.train_args, 'eval_test_at_final_loop_only', False
            ),
            metric_storing_path=f"./results/{arg_dict['dataset']}_results.csv",
            accelerator=self.accelerator,
        )
    
    def train(self):
        """Execute LSNPC training pipeline."""
        # Setup logging
        self.setup_logging()
        
        # Setup seeds
        self.setup_seeds()
        
        # Import model components early to store them
        self.model_components = self.import_model_components()
        
        # Configure model name suffixes
        if self.train_args.semi_sup:
            self.train_args.post_model += '_semi'
        if self.train_args.is_ablation:
            self.train_args.post_model += '_gauss'
        
        # Load dataset module
        data_utils = self.load_dataset_module()
        
        # Load data
        data = data_utils.load_data(self.data_args)
        n_labels = data['n_labels']
        
        # Cache n_labels to avoid reloading data in _get_n_labels()
        self._n_labels = n_labels
        
        # Verify data consistency if requested
        if getattr(self.train_args, 'checksum', False):
            from .train_utils import verify_data_consistency
            verify_data_consistency(data)
        
        # Load pretrained classifier
        encoder, emb_size = get_encoder(self.model_args.img_encoder)
        pretrained_clf = get_pretrained_model(
            self.model_args.clf_name,
            self.train_args.pretrained_clf,
            encoder,
            emb_size,
            n_labels,
        )
        
        # Create data loaders
        loaders = self.create_data_loaders(
            data,
            include_clean_loader=self.train_args.semi_sup,
        )
        
        # Training loop
        for run_index in range(self.train_args.n_repeats):
            # Generate run UID
            arg_dict = {
                **self.model_args.__dict__,
                **self.data_args.__dict__,
                **self.train_args.__dict__,
            }
            uid = generate_run_uid(arg_dict, run_index)
            arg_dict['uid'] = uid
            arg_dict['time'] = create_timestamp()
            arg_dict['run_index'] = run_index
            
            # Build model
            model = self.build_model(pretrained_clf, encoder, emb_size, n_labels)
            
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
                extra_args=arg_dict,
            )


def parse_arguments():
    """Parse command-line arguments using HuggingFace parser."""
    from transformers import HfArgumentParser
    
    parser = HfArgumentParser((
        LSNPCModelArguments,
        DataTrainingArguments,
        LSNPCTrainingArguments,
    ))
    return parser.parse_args_into_dataclasses()


def train_lsnpc():
    """Main entry point for LSNPC training."""
    # Parse arguments
    model_args, data_args, train_args = parse_arguments()
    
    # Setup logging
    from logging_utils import setup_logging
    setup_logging(
        level=train_args.logging_level.upper() if hasattr(train_args, 'logging_level') else 'INFO',
        verbose=train_args.verbose if hasattr(train_args, 'verbose') else False
    )
    
    # Log arguments line by line
    import logging
    logger = logging.getLogger(__name__)
    arg_dict = {
        **vars(model_args),
        **vars(data_args),
        **vars(train_args),
    }
    logger.info("=== Training Configuration ===")
    for key, value in sorted(arg_dict.items()):
        logger.info(f"  {key}: {value}")
    logger.info("=" * 35)
    
    # Create and run pipeline
    pipeline = LSNPCPipeline(model_args, data_args, train_args)
    pipeline.train()


if __name__ == '__main__':
    train_lsnpc()
