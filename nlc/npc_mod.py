import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.distributions.normal import Normal
from transformers import ViTModel
from typing import Dict, Tuple

from dnn.mlp import build_mlp
from dnn.utils import init_weights, TOL

torch.autograd.set_detect_anomaly(True)


class MLCEncoder(nn.Module):
    
    def __init__(
        self, 
        data_encoder: nn.Module,
        latent_dim: int, 
        n_labels: int, 
        data_emb_dim: int, 
        label_emb_dim: int, 
        n_layers: int = 4 
    ):
        super().__init__()

        self.config = {
            'latent_dim': latent_dim, 
            'n_labels': n_labels,
            'n_layers': n_layers,
            'latent_dim': latent_dim,
            'label_emb_dim': label_emb_dim,
            'data_emb_dim': data_emb_dim
        }

        self.data_encoder = data_encoder
        self.label_encoder = build_mlp(
            n_layers, n_labels, latent_dim, label_emb_dim, dp=0.1
        )
        self.label_encoder.apply(init_weights)

        self.pred_layer = nn.Linear(data_emb_dim+label_emb_dim, n_labels)

        self.norm = nn.BatchNorm1d(data_emb_dim+label_emb_dim, momentum=0.01)
 
    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        x_emb = self.data_encoder(x) 
        y_emb = self.label_encoder(y.float()) 

        emb = torch.cat((x_emb, y_emb), dim=-1)
        emb = self.norm(emb)

        y_pred = F.sigmoid(self.pred_layer(emb))  
        enc_doc = {'embedding': emb, 'y_dist': y_pred} 
        # enc_doc = {'mu': mu, 'logvar': logvar} 
        return enc_doc 

        
class MLCDecoder(nn.Module):
    
    def __init__(
        self, 
        data_encoder: nn.Module,
        latent_dim: int, 
        n_labels: int, 
        data_emb_dim: int, 
        label_emb_dim: int, 
        n_heads: int = 4,
        n_layers: int = 4
    ):
        super().__init__()

        self.config = {
            'latent_dim': latent_dim, 
            'n_labels': n_labels,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'label_emb_dim': label_emb_dim,
            'data_emb_dim': data_emb_dim
        }

        self.data_encoder = data_encoder
        self.label_encoder = build_mlp(
            n_layers, n_labels, latent_dim, label_emb_dim, dp=0.1
        )
        self.label_encoder.apply(init_weights)
 
        self.pred_layer = nn.Linear(data_emb_dim+label_emb_dim, n_labels)
        self.norm = nn.BatchNorm1d(data_emb_dim+label_emb_dim, momentum=0.01)
 
    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        x_emb = self.data_encoder(x) 
        y_emb = self.label_encoder(y.float()) 

        emb = torch.cat((x_emb, y_emb), dim=-1)
        emb = self.norm(emb)

        y_dist = F.sigmoid(self.pred_layer(emb))  
        enc_doc = {'embedding': emb, 'y_dist': y_dist} 
        return enc_doc 


class NoisyLabelCorrectionVAE(nn.Module):
    def __init__(
        self, 
        encoder: nn.Module,
        decoder: nn.Module,
        pretrained_clf: nn.Module,
        temperature: float = 1e-10,  
        use_copula: bool = True
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder 
        self.temperature = temperature

        self.pretrained_clf = pretrained_clf 
        self.use_copula = use_copula

    def forward(self, x: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        enc_doc = self.encoder(x, y_hat)
        y_dist = enc_doc['y_dist']
      
        recon_y = self.reparameterize(enc_doc) 

        recon_doc = self.decoder(x, recon_y)   
        recon_y_hat = recon_doc['y_dist'] 
        return {
            'recon_y_hat': recon_y_hat, 
            'recon_y': recon_y
        }

    def reparameterize(self, enc_doc: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.use_copula: 
            y = self.reparameterize_normal_copula(enc_doc) 
        else:
            y = self.reparameterize_bernoulli(enc_doc)
        return y

    # A variable-independent version of multivariate Bernoulli sampling
    def reparameterize_bernoulli(self, enc_doc: Dict[str, torch.Tensor]) -> torch.Tensor:
        y_dist = enc_doc['y_dist']
        u = torch.rand_like(y_dist)
        z = torch.log(y_dist) - torch.log(1-y_dist) + torch.log(u) - torch.log(1-u)
        y = F.sigmoid(z/self.temperature)
        return y
    
    # A variable-dependent version of multivariate Bernoulli sampling
    def reparameterize_normal_copula(self, enc_doc: Dict[str, torch.Tensor]) -> torch.Tensor:
        # print('reparameterize_normal_copula')
        y_dist, V, log_var = enc_doc['y_dist'], enc_doc['V'], enc_doc['log_var']
         
        var = torch.clamp(torch.exp(0.5 * log_var), min=1e-5, max=5) 

        S = torch.einsum('bdr,bcr->bdc', V, V)  
        S = S + torch.diag_embed(var) 

        L = torch.linalg.cholesky(S)
        g = torch.einsum('bcd,bd->bc', L, torch.randn_like(log_var)) 

        N = Normal(torch.zeros_like(var), var)
        u = N.cdf(g) 

        u = torch.clamp(u, min=TOL, max=0.999)
        y_dist = torch.clamp(y_dist, min=TOL, max=0.999)
        z = torch.log(y_dist) - torch.log(1-y_dist) + torch.log(u) - torch.log(1-u)
        y = F.sigmoid(z/self.temperature) 
        return y  

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
        self, 
        res_doc: Dict[str, torch.Tensor], 
        y_hat: torch.Tensor,
        y_hat_dist: torch.Tensor
    ) -> torch.Tensor: 
        recon_y_hat = res_doc['recon_y_hat']

        recon_loss = self.nll(recon_y_hat, y_hat)
 
        recon_y = torch.clamp(res_doc['recon_y'], min=0.001, max=0.999)
        y_hat_dist = torch.clamp(y_hat_dist, min=0.001, max=0.999)
        # The KL divergence between two (multivariate) Bernoulli distributions
        kl_div = torch.mean(
            recon_y * (recon_y.log() - y_hat_dist.log()) 
            + (1-recon_y) * ((1-recon_y).log() - (1-y_hat_dist).log())
        )   
        loss = recon_loss + self.beta * kl_div 
        return loss 
 

# Here is only an example
class EncoderCopulasWrapper(nn.Module):

    def __init__(self, encoder, label_emb_dim, r=3, n_layers=2):
        super().__init__()
        
        self.encoder = encoder
        
        self.config = deepcopy(encoder.config)
        self.config['label_emb_dim'] = self.config['data_emb_dim'] + label_emb_dim

        self.config['r'] = min(self.config['label_emb_dim'], r)
        self.config['n_layers'] = n_layers
         
        self.V_encoder = build_mlp(
            self.config['n_layers'], 
            self.config['label_emb_dim'], 
            self.config['latent_dim'], 
            self.config['n_labels'] * self.config['r'],
            dp=0.5,
            activation='relu'
        )
        self.log_var_encoder = build_mlp(
            self.config['n_layers'], 
            self.config['label_emb_dim'], 
            self.config['latent_dim'], 
            self.config['n_labels'],
            dp=0.1,
            activation='relu'
        )        
        self.apply(init_weights)

    def forward(self, x, y_hat):
        enc_doc = {}
        
        # 'y_dist' is contained
        enc_doc = self.encoder(x, y_hat) 

        emb = enc_doc['embedding']  

        V = F.tanh(torch.clamp(self.V_encoder(emb), max=2))     # tanh is a must to ensure the theory works  
        enc_doc['V'] = V.view(-1, self.config['n_labels'], self.config['r']) 

        log_var = torch.clamp(self.log_var_encoder(emb), max=2) 
        enc_doc['log_var'] = log_var  
        return enc_doc
    