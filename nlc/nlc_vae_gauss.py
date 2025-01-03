import math
import random 
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy 
from transformers import ViTModel
from typing import Any, Dict, Tuple

from dnn.mlp import build_mlp
from dnn.utils import init_weights, TOL
 

# tomato: 0.01
# voc: 0.01
MOMENTUM = 0.01
N_LAYERS = 4


class MlcEncoderY(nn.Module):
    
    def __init__(
        self, 
        data_encoder: nn.Module, 
        latent_dim: int, 
        n_labels: int, 
        data_emb_dim: int, 
        label_emb_dim: int, 
        nu0: int,  
        n_heads: int = 4,
        n_layers: int = 4
    ):
        super().__init__()

        self.config = {
            'latent_dim': latent_dim, 
            'n_labels': n_labels,
            'n_heads': n_heads,
            'label_emb_dim': label_emb_dim
        }

        self.nu0 = nu0 

        self.data_encoder = data_encoder
        # TEST CASE
        self.label_encoder = build_mlp(
            n_layers, n_labels, label_emb_dim, label_emb_dim, dp=0.1
        )
        self.label_encoder.apply(init_weights)

        self.mu = nn.Linear(data_emb_dim+label_emb_dim, latent_dim)
        self.logvar = nn.Linear(data_emb_dim+label_emb_dim, latent_dim) 

        self.norm = nn.BatchNorm1d(data_emb_dim+label_emb_dim, momentum=MOMENTUM) 
 
    def forward(
        self, x: torch.Tensor, y_hat: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        x_emb = self.data_encoder(x) 
        y_hat_emb = self.label_encoder(y_hat.float())
       
        emb = torch.cat((x_emb, y_hat_emb), dim=-1)
        emb = self.norm(emb)

        mu = self.mu(emb)   
        logvar = torch.clamp(self.logvar(emb), max=5)
        # nu = F.relu(self.nu(emb)) + 1

        enc_doc = {'mu': mu, 'logvar': logvar} 
        return enc_doc 


class MlcEncoderZ(nn.Module):
    
    def __init__(
        self, 
        in_dim: int, 
        latent_dim: int,
        n_layers_mu: int = N_LAYERS, 
        n_heads: int = 4
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
            n_layers_mu, in_dim, in_dim*2, in_dim, dp=0.1 
        )
        self.mu.apply(init_weights)
        self.gate = nn.Sigmoid()

        self.logvar = nn.Linear(in_dim, in_dim) 
  
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]: 
        mu = self.mu(z)
         
        logvar = torch.clamp(self.logvar(z), max=5.)  
        enc_doc = {'mu': mu, 'logvar': logvar}
        return enc_doc

        
class MlcDecoderY(nn.Module):
    
    def __init__(
        self, 
        data_encoder: nn.Module, 
        latent_dim: int, 
        n_labels: int, 
        data_emb_dim: int, 
        label_emb_dim: int, 
        n_heads: int = N_LAYERS
    ):
        super().__init__()

        self.config = {
            'latent_dim': latent_dim, 
            'n_labels': n_labels,
            'n_heads': n_heads,
            'label_emb_dim': label_emb_dim
        }

        self.data_encoder = data_encoder 
  
        self.trans = nn.Linear(data_emb_dim+latent_dim, n_labels)
 
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.BatchNorm1d(data_emb_dim+latent_dim, momentum=MOMENTUM)
 
    def forward(
        self, x: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        x_emb = self.data_encoder(x)  
        emb = torch.cat((x_emb, z), dim=-1)
        emb = self.norm(emb) 
        y_logits = self.trans(emb)   
        y = self.sigmoid(y_logits)
        return y 


class MlcDecoderZ(nn.Module):
    
    def __init__(
        self, 
        in_dim: int, 
        latent_dim: int, 
        n_layers: int = N_LAYERS
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
            n_layers, in_dim, in_dim*2, in_dim, dp=0.1
        )
        self.shift_mlp.apply(init_weights)
        self.gate = nn.Sigmoid()
  
    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        z = self.shift_mlp(z_hat)
        return z 


class NoisyLabelCorrectionVAE(nn.Module):
    def __init__(
        self, 
        encoder_y: nn.Module,
        encoder_z: nn.Module,
        decoder_y: nn.Module,
        decoder_z: nn.Module,
        pretrained_clf: nn.Module, 
        nu: int = 2 
    ):
        super().__init__()

        self.encoder_y = encoder_y
        self.encoder_z = encoder_z
        self.decoder_y = decoder_y 
        self.decoder_z = decoder_z 

        self.pretrained_clf = pretrained_clf

        self.nu = nu 

    def forward(
        self, 
        x: torch.Tensor, 
        y_hat: torch.Tensor, 
        y: torch.Tensor = None
    ) -> torch.Tensor:
        z_hat_enc_doc = self.encoder_y(x, y_hat)
        recon_z_hat = self.reparameterize_z(z_hat_enc_doc)
        
        if y is not None and random.random() < 0.5:
            z_enc_doc = self.encoder_y(x, y)
            recon_z = self.reparameterize_z(z_enc_doc)
        else:
            z_enc_doc = self.encoder_z(recon_z_hat)
            recon_z = self.reparameterize_z(z_enc_doc)
 
        z_dec_mu = self.decoder_z(recon_z)

        recon_y = self.decoder_y(x, recon_z)
        recon_y_hat =self.decoder_y(x, recon_z_hat)

        res_doc = {
            'y': recon_y,
            'y_hat': recon_y_hat, 
            'z_dec_mu': z_dec_mu,
            'z': recon_z, 
            'z_hat': recon_z_hat, 
            'z_enc_doc': z_enc_doc,
            'z_hat_enc_doc': z_hat_enc_doc
        }
        return res_doc

    def reparameterize_z(self, enc_doc: Dict[str, torch.Tensor]) -> torch.Tensor:
        mu, std = enc_doc['mu'], torch.exp(enc_doc['logvar']/2) 
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z
#
#    def reparameterize_z_hat(self, enc_doc: Dict[str, torch.Tensor]) -> torch.Tensor:
#        mu, std = enc_doc['mu'], torch.exp(enc_doc['logvar']/2) 
#        nu = self.nu # enc_doc['nu'] # 
#        std = torch.clamp(std, min=1e-4) 
#        # It may return NaNs on GPUs or M1s without using CPUs
#        T = D.studentT.StudentT(df=nu, loc=mu.cpu(), scale=std.cpu())
#        z_hat = T.rsample()
#        return z_hat.to(mu.device)
  
    @torch.no_grad()
    def sample(self, x: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        y_enc = self.encoder(x, y_hat)
        recon_y = self.reparameterize(y_enc)
        return recon_y 


class CorrectionLoss(nn.Module):
    
    def __init__(self, beta: float):
        super().__init__()
        
        self.beta = beta
        self.nll = nn.BCELoss(reduction='mean')

    def forward(
        self, res_doc: Dict[str, Any], y_hat: torch.Tensor
    ) -> torch.Tensor:
        ''' recon_pair: a tuple containing recon_y_hat and y_hat
            kl_pair: a tuple containing y_enc and y_prior
        ''' 
        recon_y_hat = res_doc['y_hat']
        y_hat = torch.clamp(y_hat, min=TOL)
        recon_loss = self.nll(recon_y_hat, y_hat) # focal_loss(recon_y_hat, y_hat) 

        recon_z, recon_z_hat = res_doc['z'], res_doc['z_hat']
        mu, std = (
            res_doc['z_hat_enc_doc']['mu'],
            torch.exp(res_doc['z_hat_enc_doc']['logvar']/2)
        )
        z_mu = res_doc['z_dec_mu'] 
        z_hat_kl_div = torch.mean(
            - D.normal.Normal(
                  torch.zeros_like(mu), torch.ones_like(std)
              ).log_prob(recon_z_hat - z_mu)
            + D.normal.Normal(mu, std).log_prob(recon_z_hat)
        )
        
        mu, logvar = res_doc['z_enc_doc']['mu'], res_doc['z_enc_doc']['logvar']
        z_kl_div = - 0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        loss = recon_loss + self.beta * (z_hat_kl_div + z_kl_div) 
        return loss 


class SemiSupCorrectionLoss(CorrectionLoss):
    
    def __init__(self, beta: float, lbd: float) -> torch.Tensor:
        
        super().__init__(beta)
        
        self.lbd = lbd  # lambda 

    def forward(
        self,
        recon_pair: Tuple[torch.Tensor, torch.Tensor],
        kl_pair: Tuple[torch.Tensor, torch.Tensor],
        val_pair: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        loss = super().forward(recon_pair, kl_pair)

        clean_y_pred, clean_y_true = val_pair
        clean_y_pred = torch.clamp(clean_y_pred, min=TOL)
        loss += self.lbd * F.cross_entropy(
            clean_y_pred, clean_y_true, reduction='mean'
        )
        return loss 

 