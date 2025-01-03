import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel


class MultilabelClassifier(nn.Module):

    def __init__(self, encoder, emb_size, n_labels, dp=0.1):
        super(MultilabelClassifier, self).__init__()
        self.encoder = encoder 
        self.dp = nn.Dropout(dp)
        self.cls_layer = nn.Linear(emb_size, n_labels) 
        
        nn.init.xavier_uniform_(self.cls_layer.weight)
        self.pred_sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.encoder(x)
        y = self.dp(y) 
        y = self.cls_layer(y) 
        return y


class ViTModelWrapper(nn.Module):

    def __init__(self, vit):
        super(ViTModelWrapper, self).__init__()
        self.vit = vit

    def forward(self, x):
        output = self.vit(x)
        pooler_output = output.pooler_output
        
        if pooler_output is None:
            raise ValueError('Pooler output from ViTModel is None')
        else:
            return pooler_output
        
