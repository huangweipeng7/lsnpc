import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from packaging import version


TOL = 1e-10     # For numerical stability


def init_weights(m): 
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


def freeze_param(m):
    m.requires_grad = False


def get_device():
    if (
        version.parse(torch.__version__) > version.parse('1.12.0') 
        and torch.backends.mps.is_available()
    ):
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return torch.device(device)
