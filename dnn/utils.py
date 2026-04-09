import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from packaging import version


TOL = 1e-10     # For numerical stability


def init_weights(m, nonlinearity='gelu'):
    """Initialize weights using appropriate strategy based on module type and activation.
    
    This function implements architecture-aware initialization with different strategies
    for encoder/decoder components and different activation functions.
    
    Args:
        m: PyTorch module to initialize
        nonlinearity: Activation function type ('relu', 'leaky_relu', 'sigmoid', 
                     'tanh', 'linear'). Default: 'gelu' (used in this project's MLPs)
    
    Note:
        - Only initializes modules that are submodules of encoder or decoder components
        - Uses Kaiming/He initialization for ReLU-based networks (better for deep networks)
        - Uses Xavier/Glorot initialization for Sigmoid/Tanh activations
        - Biases are initialized to zero
    """
    if isinstance(m, nn.Linear):
        # Check if module is part of encoder or decoder by examining parent module names
        is_encoder_decoder = False
        for name, _ in m.named_modules():
            if name:  # Skip the module itself
                name_lower = name.lower()
                if 'encoder' in name_lower or 'decoder' in name_lower:
                    is_encoder_decoder = True
                    break
        
        # Only initialize if this is a submodule of encoder or decoder
        if not is_encoder_decoder:
            return
        
        # Apply Kaiming/He initialization (better for modern activations)
        # For very deep networks or ReLU variants, this often outperforms Xavier
        if nonlinearity.lower() in ['relu', 'leaky_relu', 'gelu', 'silu']:
            nn.init.kaiming_uniform_(
                m.weight, a=0, mode='fan_in', 
                nonlinearity=nonlinearity.lower()
            )
        else:
            # Calculate appropriate gain based on activation function
            gain = nn.init.calculate_gain(nonlinearity)
            # Use Xavier/Glorot for sigmoid/tanh/linear
            nn.init.xavier_uniform_(m.weight, gain=gain)
        
        # Initialize bias to zero
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def init_weights_xavier(m, gain=1.0):
    """Legacy Xavier initialization for backward compatibility.
    
    Args:
        m: PyTorch module to initialize
        gain: Scaling factor for Xavier initialization (default: 1.0)
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


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
