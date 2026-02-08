import torch
import torch.nn as nn
import torch.nn.functional as F
 

def build_mlp(
    n_layers, 
    in_dim, 
    latent_dim,  
    out_dim,  
    activation='gelu', 
    norm=None,
    dp=0.1
):
    ''' The supported activation functions are only GeLU and ReLU.
    # '''
    layers = []
    for i in range(n_layers-1):
        layers.append(
            nn.Linear(in_dim if i == 0 else latent_dim, latent_dim)
        )

        if dp > 0:
            layers.append(nn.Dropout(dp))

        if norm == 'batchnorm':
            layers.append(nn.BatchNorm1d(latent_dim))
        elif norm == 'layernorm':
            layers.append(nn.LayerNorm(latent_dim))

        layers.append(
            nn.GELU() if activation == 'gelu' else nn.ReLU()
        )

    layers.append(nn.Linear(latent_dim, out_dim))

    return nn.Sequential(*layers)

    # return RMLP(n_layers, in_dim, latent_dim, out_dim, norm, dp)
 

class RMLP(nn.Module): 

    def __init__(self, n_layers, in_dim, latent_dim, out_dim, norm=None, dp=0.1):
        super(RMLP, self).__init__()

        layers = []
        for i in range(n_layers-1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, latent_dim),
                    nn.Dropout(dp),
                    nn.GELU(),
                    nn.Linear(latent_dim, in_dim)
                )
            )
        self.models = nn.ModuleList(layers)
        self.last_layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        for model in self.models:
            x = F.normalize(x + model(x)) 
            x =- F.gelu(x)
        y = self.last_layer(x)
        return y 
