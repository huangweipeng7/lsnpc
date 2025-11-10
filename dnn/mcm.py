import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Tuple
from torch import Tensor
from transformers import ViTModel
from utils import ConstraintUtils

class MCMClassifier(nn.Module):

    def __init__(self, encoder, emb_size, n_labels, dp=0.1):
        super(MCMClassifier, self).__init__()
        self.encoder = encoder 
        self.dp = nn.Dropout(dp)
        self.cls_layer = nn.Linear(emb_size, n_labels) 
        
        nn.init.xavier_uniform_(self.cls_layer.weight)
        self.pred_sigmoid = nn.Sigmoid()

        # Initialize confusion matrices A and B as learnable parameters
        # A: confusion matrix for positive class influence
        # B: confusion matrix for negative class influence
        self.A = nn.Parameter(torch.eye(n_labels), requires_grad=True)
        self.B = nn.Parameter(torch.zeros(n_labels, n_labels), requires_grad=True)
        # nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)
        # Initialize constraint utilities for maintaining probability constraints
        self.constraint_utils = ConstraintUtils()


    def forward(self, x):
        # Compute the base predictions
        y = self.encoder(x)
        y = self.dp(y) 
        y = self.cls_layer(y)  
        clean_probs = self.pred_sigmoid(y)

        assert not torch.any(torch.isnan(clean_probs))

        self._validate_probabilities(clean_probs, "Base predictions y")

        # Compute noisy probabilities using the MCM model equation
        # g(x) = A*f(x) + B*(1-f(x))
        noisy_probs = self.compute_noisy_probs(clean_probs)

        # Validate noisy probabilities are in valid range
        self._validate_probabilities(noisy_probs, "noisy_probs")
 
        return y, noisy_probs
    

    def compute_noisy_probs(self, clean_probs: Tensor) -> Tensor:
        """
        Compute noisy probabilities from clean probabilities using confusion matrices.
        
        Implements Equation (4) from the paper:
        g_n = A * f(x_n) + B * (1 - f(x_n))
        
        Args:
            clean_probs: Clean probability predictions f(x) of shape (batch_size, num_classes)
            
        Returns:
            Noisy probability predictions g(x) of shape (batch_size, num_classes)
        """
        batch_size = clean_probs.shape[0]
        
        # Transpose clean_probs to (num_classes, batch_size) for matrix multiplication
        clean_probs_t = clean_probs.T  # Shape: (num_classes, batch_size)
        
        assert not torch.any(torch.isnan(clean_probs_t))

        # Compute A * f(x) using matrix multiplication
        # A shape: (num_classes, num_classes), clean_probs_t shape: (num_classes, batch_size)
        # Result shape: (num_classes, batch_size)
        term_A = torch.matmul(self.A, clean_probs_t)
        
        assert not torch.any(torch.isnan(term_A))

        # Compute B * (1 - f(x)) using matrix multiplication
        complement_probs = 1.0 - clean_probs_t  # Shape: (num_classes, batch_size)
        term_B = torch.matmul(self.B, complement_probs)
        
        assert not torch.any(torch.isnan(term_B))
 
        # Combine terms: g = A*f(x) + B*(1-f(x))
        noisy_probs_t = term_A + term_B  # Shape: (num_classes, batch_size)
        
        # Transpose back to (batch_size, num_classes)
        noisy_probs = noisy_probs_t.T
        
        # Ensure numerical stability by clipping to valid probability range
        noisy_probs = torch.clamp(noisy_probs, min=0.0, max=1.0)
        
        return noisy_probs
    

    def apply_constraints(self) -> None:
        """
        Apply probability constraints to confusion matrices A and B.
        
        Ensures that the concatenated matrix [A|B] satisfies:
        - All entries are non-negative (A ≥ 0, B ≥ 0)
        - Each row sums to 1 (probability distribution constraint)
        
        This should be called after each optimizer step during training.
        """
        # Apply constraints using the constraint utilities
        A_projected, B_projected = self.constraint_utils.project_confusion_matrices(
            self.A, self.B
        )
        
        # Update the parameters with projected values
        # Use in-place operations to maintain gradient tracking
        with torch.no_grad():
            self.A.data = A_projected
            self.B.data = B_projected
        

    def _validate_probabilities(self, probs: Tensor, name: str) -> None:
        """
        Validate that probability tensors are in valid range [0,1].
        
        Args:
            probs: Probability tensor to validate
            name: Name of the tensor for error messages
            
        Raises:
            ValueError: If probabilities are outside valid range
        """ 
        assert not torch.any(torch.isnan(probs))

        if torch.any(probs < 0) or torch.any(probs > 1):
            min_val = probs.min().item()
            max_val = probs.max().item()
            raise ValueError(
                f"{name} must be in range [0,1], got [{min_val:.3f}, {max_val:.3f}]"
            )
        

class MCMLoss(nn.Module):
    def __init__(self, p: float = 0.2, sparsity_lambda: float = 1e-3):
        super(MCMLoss, self).__init__()
        self.p = p  # Quasi-norm parameter for sparsity
        self.epsilon = 1e-8  # Small constant for numerical stability
        self.sparsity_lambda = sparsity_lambda
        self.bce_loss = nn.BCELoss()
    
    def compute_sparsity_loss(self, clean_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute sparsity regularization loss using ℓ_p quasi-norm.
        
        Implements: L_S(f) = 1/N * Σ_n ||f(x_n)||_p^p
        where ||u||_p^p = Σ_i (|u_i| + ε)^p for numerical stability.
        
        Args:
            clean_probs: Clean probability predictions of shape (batch_size, num_classes)
            
        Returns:
            Sparsity regularization loss value
        """
        # Compute ℓ_p quasi-norm: sum over classes of (|value| + ε)^p
        # Add epsilon for numerical stability and to avoid gradient issues at 0
        sparsity_terms = torch.pow(torch.abs(clean_probs) + self.epsilon, self.p)
        
        # Sum over classes and average over batch
        sparsity_loss = torch.mean(torch.sum(sparsity_terms, dim=1)) 
        return sparsity_loss
    
    def forward(self, 
                noisy_predictions: torch.Tensor,
                noisy_targets: torch.Tensor,
                clean_probs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss including BCE and sparsity regularization.
        
        Args:
            noisy_predictions: Predicted noisy probabilities g(x)
            noisy_targets: Observed noisy labels
            clean_probs: Clean probability predictions f(x)
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Compute BCE loss on noisy predictions
        # bce_loss = self.compute_bce_loss(clean_probs, noisy_targets)
        bce_loss = self.bce_loss(clean_probs, noisy_targets)        

        # Compute sparsity regularization loss
        sparsity_loss = self.compute_sparsity_loss(clean_probs)
        
        # Combine losses
        total_loss = bce_loss + self.sparsity_lambda * sparsity_loss
        
        # Prepare loss components for logging
        loss_components = {
            'total_loss': total_loss.item(),
            'bce_loss': bce_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'sparsity_lambda': self.sparsity_lambda
        }
        # print('Loss components:', loss_components)
        
        return total_loss, loss_components


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
        
