import torch
import torch.distributions as D
import torch.nn as nn
from random import random
from typing import Any, Dict, Tuple

from dnn.gat import LabelEncoderGAT
from dnn.mlp import build_mlp
from dnn.utils import init_weights, TOL


# =============================================================================
# Configuration Constants: MOMENTUM for batch normalization
# =============================================================================
# tomato: 0.01
# voc: 0.01
# coco: 
MOMENTUM = 0.1
N_LAYERS = 2


# =============================================================================
# Encoder Models
# =============================================================================
class MlcEncoderY(nn.Module):
    """Encoder for data and label embeddings to latent space."""
    
    def __init__(
        self, 
        data_encoder: nn.Module, 
        latent_dim: int, 
        n_labels: int, 
        data_emb_dim: int, 
        label_emb_dim: int, 
        nu0: int,  
        dp: float = 0.1
    ):
        super().__init__()

        self.config = {
            'latent_dim': latent_dim, 
            'n_labels': n_labels,
            'label_emb_dim': label_emb_dim,
            'dp': dp
        }

        self.nu0 = nu0 

        # Data encoding pathway
        self.data_encoder = data_encoder 
        self.data_fc = nn.Linear(data_emb_dim, data_emb_dim)

        # Label encoding using GAT
        self.label_encoder = LabelEncoderGAT(
            n_labels, latent_dim, 
            [label_emb_dim*2, label_emb_dim], 
            pooling='mean'
        )

        # Latent space parameters
        self.mu = nn.Linear(data_emb_dim+label_emb_dim, latent_dim)
        self.logvar = nn.Linear(data_emb_dim+label_emb_dim, latent_dim) 

        # Normalization
        # self.norm = nn.BatchNorm1d(data_emb_dim+label_emb_dim, momentum=MOMENTUM) 
        self.norm = nn.RMSNorm(data_emb_dim+label_emb_dim)

    def forward(
        self, x: torch.Tensor, y_hat: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        x_emb = self.data_encoder(x) 
        x_emb = self.data_fc(x_emb) 

        y_hat_emb, _ = self.label_encoder(y_hat.round().int())
       
        emb = torch.cat((x_emb, y_hat_emb), dim=-1)
        emb = self.norm(emb)

        mu = self.mu(emb)   
        logvar = torch.clamp(self.logvar(emb), max=5)
        # nu = F.relu(self.nu(emb)) + 1

        enc_doc = {'mu': mu, 'logvar': logvar}#, 'nu': nu} 
        return enc_doc 


class MlcEncoderZ(nn.Module):
    """Encoder for latent variable z."""
    
    def __init__(
        self, 
        in_dim: int, 
        latent_dim: int,
        n_layers_mu: int = N_LAYERS, 
        dp: float = 0.1
    ):
        super().__init__()

        out_dim = in_dim  
        self.config = {
            'n_layers_mu': n_layers_mu, 
            'in_dim': in_dim, 
            'latent_dim': latent_dim,                                                                
            'out_dim': out_dim 
        }
  
        self.mu = build_mlp(
            n_layers_mu, in_dim, latent_dim, in_dim, dp=dp, norm='rmsnorm'
        )  
        self.logvar = nn.Linear(in_dim, in_dim) 

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]: 
        mu = self.mu(z)  
        logvar = torch.clamp(self.logvar(z), max=5)   
        return {'mu': mu, 'logvar': logvar}


# =============================================================================
# Decoder Models
# =============================================================================
class MlcDecoderY(nn.Module):
    """Decoder from latent space to label predictions."""
    
    def __init__(
        self, 
        data_encoder: nn.Module, 
        latent_dim: int, 
        n_labels: int, 
        data_emb_dim: int, 
        label_emb_dim: int, 
        dp: float = 0.1
    ):
        super().__init__()

        self.config = {
            'latent_dim': latent_dim, 
            'n_labels': n_labels,
            'label_emb_dim': label_emb_dim
        }

        self.data_encoder = data_encoder 
        self.data_fc = nn.Linear(data_emb_dim, data_emb_dim)
  
        self.trans = nn.Linear(data_emb_dim+latent_dim, n_labels) 

        self.sigmoid = nn.Sigmoid()
        # self.norm = nn.BatchNorm1d(data_emb_dim+latent_dim, momentum=MOMENTUM)    
        self.norm = nn.RMSNorm(data_emb_dim+latent_dim)
 
    def forward(
        self, x: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        x_emb = self.data_encoder(x)  
        x_emb = self.data_fc(x_emb)
        emb = torch.cat((x_emb, z), dim=-1)
        emb = self.norm(emb) 
        y_logits = self.trans(emb)    
        return y_logits 


class MlcDecoderZ(nn.Module):
    """Decoder for latent variable shift."""
    
    def __init__(
        self, 
        in_dim: int, 
        latent_dim: int, 
        n_layers: int = N_LAYERS,
        dp: float = 0.1
    ):
        super().__init__()

        out_dim = in_dim 
        self.config = {
            'in_dim': in_dim, 
            'latent_dim': latent_dim,
            'n_layers': n_layers,  
            'out_dim': out_dim 
        }
   
        self.shift_mlp = build_mlp(
            n_layers, in_dim, latent_dim, in_dim, dp=dp, norm='rmsnorm'
        )
        # self.shift_mlp.apply(init_weights)
        self.gate = nn.Sigmoid()
  
    def forward(self, z_hat: torch.Tensor) -> torch.Tensor: 
        z = self.shift_mlp(z_hat)
        return z 


# =============================================================================
# Main VAE Model
# =============================================================================
class NoisyLabelCorrectionVAE(nn.Module):
    """Variational Autoencoder for Noisy Label Correction."""
    
    def __init__(
        self, 
        encoder_y: nn.Module,
        encoder_z: nn.Module,
        decoder_y: nn.Module,
        decoder_z: nn.Module,
        pretrained_clf: nn.Module, 
        nu: int = 2,
        eta: float = 0.5
    ):
        super().__init__()

        self.encoder_y = encoder_y
        self.encoder_z = encoder_z
        self.decoder_y = decoder_y 
        self.decoder_z = decoder_z 
  
        self.pretrained_clf = pretrained_clf

        self.nu = nu
        self.eta = eta 

    def forward(
        self, 
        x: torch.Tensor, 
        y_hat: torch.Tensor, 
        y: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        # Encode noisy predictions
        z_hat_enc_doc = self.encoder_y(x, y_hat)
        recon_z_hat = self.reparameterize_z_hat(z_hat_enc_doc) 
        
        # Semi-supervised learning: use ground truth with probability eta
        if y is not None and random() < self.eta:
            z_enc_doc = self.encoder_y(x, y)
        else:
            z_enc_doc = self.encoder_z(recon_z_hat)
        recon_z = self.reparameterize_z(z_enc_doc)
 
        # Decode latent variables
        z_dec_mu = self.decoder_z(recon_z)

        # Generate predictions
        recon_y_logits = self.decoder_y(x, recon_z)
        recon_y_hat_logits = self.decoder_y(x, recon_z_hat)

        res_doc = {
            'y_logits': recon_y_logits,
            'y_hat_logits': recon_y_hat_logits, 
            'z_dec_mu': z_dec_mu,
            'z': recon_z, 
            'z_hat': recon_z_hat, 
            'z_enc_doc': z_enc_doc,
            'z_hat_enc_doc': z_hat_enc_doc
        }
        return res_doc

    def reparameterize_z(self, enc_doc: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Standard Gaussian reparameterization trick."""
        mu, std = enc_doc['mu'], torch.exp(enc_doc['logvar']/2) 
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def reparameterize_z_hat(self, enc_doc: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Student-T reparameterization for noisy predictions."""
        mu, std = enc_doc['mu'], torch.exp(enc_doc['logvar']/2) 
        nu = self.nu # enc_doc['nu'] # 
        std = torch.clamp(std, min=1e-4)  

        # device = mu.device 
        # T = D.StudentT(
        #     df=nu if not isinstance(nu, torch.Tensor) else nu.to(dtype=torch.float32, device='cpu'), 
        #     loc=mu.to(dtype=torch.float32, device='cpu'), 
        #     scale=std.to(dtype=torch.float32, device='cpu')
        # )  
        z_hat = D.StudentT(df=nu, loc=mu, scale=std).rsample()  
        return z_hat 
    
    @torch.no_grad()
    def sample(self, x: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        """Sample corrected labels from the model.
        
        Args:
            x: Input data tensor
            y_hat: Noisy label predictions
            
        Returns:
            Corrected label logits
        """
        # Encode noisy predictions to get z_hat
        z_hat_enc_doc = self.encoder_y(x, y_hat)
        recon_z_hat = self.reparameterize_z_hat(z_hat_enc_doc)
        
        # Encode through encoder_z to get clean latent z
        z_enc_doc = self.encoder_z(recon_z_hat)
        recon_z = self.reparameterize_z(z_enc_doc)
        
        # Decode to get corrected predictions
        recon_y_logits = self.decoder_y(x, recon_z)
        
        return recon_y_logits


# =============================================================================
# Loss Functions
# =============================================================================
class CorrectionLoss(nn.Module):
    """Loss function for noisy label correction."""
    
    def __init__(self, beta: float, nu: int = 2, nu0: int = 2):
        super().__init__()
        
        self.beta = beta
        self.nu0 = nu0
        self.nu = nu 
        self.bce_logits = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(
        self, 
        res_doc: Dict[str, Any], 
        y_hat: torch.Tensor,
        y_true: torch.Tensor = None 
    ) -> torch.Tensor:
        ''' recon_pair: a tuple containing recon_y_hat and y_hat
            kl_pair: a tuple containing y_enc and y_prior
        ''' 
        # Reconstruction loss
        recon_y_hat_logits = res_doc['y_hat_logits']
        recon_loss = self.bce_logits(recon_y_hat_logits, y_hat)  
        if y_true is not None:
            recon_loss += self.bce_logits(res_doc['y_logits'], y_true)  

        # MC approximation of KL divergence for z_hat (Student-T)
        mu, std, nu = (
            res_doc['z_hat_enc_doc']['mu'], 
            torch.exp(res_doc['z_hat_enc_doc']['logvar']/2),
            self.nu # res_doc['z_hat_enc_doc']['nu'] #
        ) 
        # z_hat_kl_div = torch.mean(
        #     - D.StudentT(self.nu0).log_prob(recon_z_hat - z_mu) 
        #     + D.StudentT(nu, mu, std).log_prob(recon_z_hat)
        # )

        recon_z_hat = res_doc['z_hat']
        z_mu = res_doc['z_dec_mu'] 
        
        logp = D.StudentT(self.nu0).log_prob(recon_z_hat - z_mu) 
        logq = D.StudentT(nu, mu, std).log_prob(recon_z_hat)
        log_p_over_q = logp - logq
        z_hat_kl_div = - torch.mean(log_p_over_q.exp() - 1. - log_p_over_q)

        # KL divergence for z (Gaussian)
        mu, logvar = res_doc['z_enc_doc']['mu'], res_doc['z_enc_doc']['logvar']
        z_kl_div = - 0.5 * torch.mean(1. + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        return recon_loss + self.beta * (z_hat_kl_div + z_kl_div) 
 