"""Multi-Layer Perceptron (MLP) implementations for LSNPC project.

This module provides flexible MLP architectures with various configurations
including different activation functions, normalization layers, and dropout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import Literal, Optional


SUPPORTED_NORMS = {'batchnorm', 'layernorm', 'rmsnorm'}


def build_mlp(
    n_layers: int,
    in_dim: int,
    latent_dim: int,
    out_dim: int,
    activation: Literal['gelu', 'relu'] = 'gelu',
    norm: Optional[Literal['batchnorm', 'layernorm', 'rmsnorm']] = None,
    dp: float = 0.1
) -> nn.Sequential:
    """Build a multi-layer perceptron with customizable architecture.
    
    Args:
        n_layers: Total number of layers (including input and output).
        in_dim: Input dimension.
        latent_dim: Hidden layer dimension.
        out_dim: Output dimension.
        activation: Activation function ('gelu' or 'relu'). Default: 'gelu'.
        norm: Normalization type ('batchnorm', 'layernorm', 'rmsnorm', or None).
        dp: Dropout rate. Set to 0 to disable. Default: 0.1.
        
    Returns:
        PyTorch Sequential model with the specified architecture.
        
    Example:
        >>> mlp = build_mlp(n_layers=3, in_dim=512, latent_dim=256, out_dim=80)
        >>> x = torch.randn(32, 512)
        >>> output = mlp(x)
    """
    # Warn if an unsupported norm is specified
    if norm is not None and norm not in SUPPORTED_NORMS:
        warnings.warn(
            f"Unsupported norm type '{norm}'. "
            f"Supported norms: {SUPPORTED_NORMS}. No normalization will be applied.",
            UserWarning
        )
        norm = None
    
    layers = []
    
    for i in range(n_layers - 1):
        # Linear layer with dynamic input dimension
        input_dim = in_dim if i == 0 else latent_dim
        layers.append(nn.Linear(input_dim, latent_dim))

        # Dropout (if enabled)
        if dp > 0:
            layers.append(nn.Dropout(dp))

        # Normalization (if specified)
        if norm == 'batchnorm':
            layers.append(nn.BatchNorm1d(latent_dim))
        elif norm == 'layernorm':
            layers.append(nn.LayerNorm(latent_dim))
        elif norm == 'rmsnorm':
            layers.append(nn.RMSNorm(latent_dim))

        # Activation function
        if activation == 'gelu':
            layers.append(nn.GELU())
        else:
            layers.append(nn.ReLU())

    # Output layer
    layers.append(nn.Linear(latent_dim, out_dim))

    return nn.Sequential(*layers)


class RMLP(nn.Module):
    """Residual Multi-Layer Perceptron with skip connections.
    
    This implementation uses residual blocks with normalization and 
    GeLU activation in each block.
    
    Args:
        n_layers: Number of residual blocks.
        in_dim: Input dimension.
        latent_dim: Hidden dimension within each block.
        out_dim: Output dimension.
        norm: Whether to apply normalization (currently disabled).
        dp: Dropout rate. Default: 0.1.
    """

    def __init__(
        self,
        n_layers: int,
        in_dim: int,
        latent_dim: int,
        out_dim: int,
        norm: Optional[str] = None,
        dp: float = 0.1
    ):
        super().__init__()

        # Build residual blocks
        layers = []
        for i in range(n_layers - 1):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections.
        
        Args:
            x: Input tensor of shape (batch_size, in_dim).
            
        Returns:
            Output tensor of shape (batch_size, out_dim).
        """
        for model in self.models:
            # Residual connection with normalization
            x = F.normalize(x + model(x))
            x = F.gelu(x)
        
        y = self.last_layer(x)
        return y