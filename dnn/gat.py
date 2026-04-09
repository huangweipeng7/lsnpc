import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """
    Single Graph Attention Layer for label encoding.
    Implements the GAT mechanism for a sparse graph of labels.
    """
    def __init__(self, num_labels, in_features: int, out_features: int, num_heads: int = 2,
                 dropout: float = 0.1, concat: bool = True, add_self_loops: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.add_self_loops = add_self_loops
        self.num_labels = num_labels

        # Multi-head feature transformation
        self.W = nn.Parameter(torch.empty((num_heads, in_features, out_features)))
        
        # Attention mechanism parameters
        self.attn_src = nn.Parameter(torch.empty((num_heads, out_features, 1)))
        self.attn_dst = nn.Parameter(torch.empty((num_heads, out_features, 1)))
        
        # Bias and dropout
        self.bias = nn.Parameter(torch.empty((num_labels, num_heads, out_features)))
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters following GAT paper conventions."""
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        nn.init.constant_(self.bias, 0)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_labels, in_features] - Label embeddings
            adj: [batch_size, num_labels, num_labels] - Label adjacency matrix
            
        Returns:
            [batch_size, num_labels, out_features*num_heads] if concat=True
            [batch_size, num_labels, out_features] otherwise
        """
        batch_size, num_labels, _ = x.shape
        
        if adj is None:
            adj = torch.ones(batch_size, num_labels, num_labels, device=x.device)
        
        # Add self-loops if specified
        if self.add_self_loops:
            adj = adj + torch.eye(num_labels, device=adj.device).unsqueeze(0)
        
        # Apply dropout to adjacency
        adj = self.dropout(adj)
        
        # Linear transformation: [batch, num_labels, in] -> [batch, num_labels, heads, out]
        h = torch.einsum('bni,hio->bnho', x, self.W)

        # Compute attention scores: [batch, heads, num_labels, num_labels]
        attention = torch.einsum('bnhd,bmhd->bhnm', h, h)
        attention = attention / (self.out_features ** 0.5)
        attention = F.leaky_relu(attention, negative_slope=0.2)

        # Apply adjacency mask
        adj_mask = adj.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        attention = attention.masked_fill(adj_mask == 0, -1e9)

        # Softmax and dropout
        attention = self.dropout(F.softmax(attention, dim=-1))

        # Apply attention and transpose: [batch, heads, num_labels, out]
        h = h.transpose(1, 2)
        h = torch.einsum('bhnm,bhno->bhmo', attention, h)
        h = h.transpose(1, 2)  # [batch, num_labels, heads, out]

        # Add bias
        h = h + self.bias.unsqueeze(0)

        # Combine heads
        if self.concat and self.num_heads > 1:
            out = h.reshape(batch_size, num_labels, -1)
        else:
            out = h.mean(dim=2)

        return F.elu(out)


class MultiHeadGAT(nn.Module):
    """
    Multi-layer Graph Attention Network for binary label encoding.
    """
    def __init__(self, num_labels, 
                 in_features: int, hidden_features: list, num_heads: list,
                 dropout: float = 0.1, residual: bool = True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.residual = residual
        self.num_labels = num_labels
        
        # Create GAT layers
        for i, (hidden, heads) in enumerate(zip(hidden_features, num_heads)):
            in_dim = in_features if i == 0 else hidden_features[i-1]
            if i > 0 and num_heads[i-1] > 1:
                in_dim = in_dim * num_heads[i-1] if i < len(hidden_features) else in_dim
            
            self.layers.append(
                GATLayer(
                    num_labels=num_labels,
                    in_features=in_dim,
                    out_features=hidden,
                    num_heads=heads,
                    dropout=dropout,
                    concat=True
                )
            )
        
        # Final projection
        final_in = hidden_features[-1] 
        if num_heads[-1] > 1 and len(hidden_features) > 1:
            final_in = final_in * num_heads[-1]
        self.projection = nn.Linear(final_in, hidden_features[-1])
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_labels, in_features] - Input label embeddings
            adj: [batch_size, num_labels, num_labels] - Optional adjacency matrix
            
        Returns:
            [batch_size, num_labels, hidden_features[-1]] - Encoded label embeddings
        """
        h = x
        
        for i, layer in enumerate(self.layers):
            # Save residual if needed
            if self.residual and i > 0:
                residual = h
            
            # Apply GAT layer
            h = layer(h, adj)

            # Apply residual connection (skip connection)
            if self.residual and i > 0 and h.shape == residual.shape:
                h = h + residual
    
        # Final projection
        h = self.projection(h)
        
        return h


class LearnableAdjacencyGAT(nn.Module):
    """
    GAT with learnable adjacency matrix for variable number of tokens.
    Supports CLS token when specified.
    """
    def __init__(self, base_num_labels: int, embedding_dim: int, hidden_dims: list,
                 num_heads: list = [4, 4], use_cls_token: bool = False,
                 sparsity_threshold: float = 0.1, temperature: float = 1.0):
        super().__init__()
        self.base_num_labels = base_num_labels
        self.use_cls_token = use_cls_token
        self.num_tokens = base_num_labels + (1 if use_cls_token else 0)
        self.sparsity_threshold = sparsity_threshold
        self.temperature = temperature
        
        # Learnable adjacency logits for all tokens (including CLS if used)
        self.adj_logits = nn.Parameter(
            torch.randn(self.num_tokens, self.num_tokens)
        )
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i, (hidden, heads) in enumerate(zip(hidden_dims, num_heads)):
            in_dim = embedding_dim if i == 0 else hidden_dims[i-1] * num_heads[i-1]
            concat = True if i < len(hidden_dims) - 1 else False
            
            self.gat_layers.append(GATLayer(
                num_labels=self.num_tokens,
                in_features=in_dim,
                out_features=hidden,
                num_heads=heads,
                dropout=0.1,
                concat=concat
            ))
            
            # Layer normalization
            self.gat_layers.append(nn.LayerNorm(hidden * heads if concat else hidden))
        
        # Final projection
        final_dim = hidden_dims[-1]
    
        self.projection = nn.Linear(final_dim, hidden_dims[-1])
        
        # Optional: Initialize with prior knowledge
        self.register_buffer('prior_adj', None)
    
    def set_prior_adjacency(self, prior_adj: torch.Tensor):
        """Set prior adjacency matrix."""
        if self.use_cls_token:
            # Expand prior_adj to include CLS token
            expanded_prior = torch.zeros(
                self.num_tokens, self.num_tokens, 
                device=prior_adj.device, dtype=prior_adj.dtype
            )
            # Original labels
            expanded_prior[1:, 1:] = prior_adj
            # CLS token connected to all other tokens (bidirectional)
            expanded_prior[0, :] = 1.0
            expanded_prior[:, 0] = 1.0
            self.prior_adj = expanded_prior
        else:
            self.prior_adj = prior_adj
    
    def get_adjacency(self, batch_size: int = 1) -> torch.Tensor:
        """
        Get adjacency matrix with sparsity control.
        """
        logits = self.adj_logits
        
        # Incorporate prior knowledge if available
        if self.prior_adj is not None:
            logits = logits + self.prior_adj
        
        # Differentiable sparsification
        if self.training:
            # Gumbel-Softmax for differentiable sampling
            uniform = torch.rand_like(logits)
            gumbel = -torch.log(-torch.log(uniform + 1e-8) + 1e-8)
            y = torch.sigmoid((logits + gumbel) / self.temperature)
        else:
            # Hard threshold at inference
            y = torch.sigmoid(logits)
        
        # Apply sparsity threshold
        adj = (y > self.sparsity_threshold).float()
        
        # Ensure no self-loops in adjacency (they're added in GAT layer)
        adj = adj.fill_diagonal_(0)
        
        # Expand to batch dimension
        adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
        
        return adj
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [batch_size, num_tokens, embedding_dim] - Token embeddings
            
        Returns:
            [batch_size, num_tokens, hidden_dims[-1]] - Encoded tokens
            torch.Tensor: Learned adjacency matrix for visualization
        """
        batch_size = x.shape[0]
        
        # Get learnable adjacency
        adj = self.get_adjacency(batch_size)
        
        # Apply GAT layers with residual connections
        h = x
        prev_h = None

        for i, layer in enumerate(self.gat_layers):
            if isinstance(layer, GATLayer):
                # Apply GAT layer
                h_new = layer(h, adj)

                # Apply residual connection if dimensions match
                if prev_h is not None and h_new.shape == prev_h.shape:
                    h_new = h_new + prev_h
                h = h_new
            else:
                # Apply layer normalization
                h = layer(h)

            if isinstance(layer, GATLayer):
                prev_h = h.detach()

        # Final projection
        h = self.projection(h)

        return h, adj.detach() 


class LabelEncoderGAT(nn.Module):
    """
    Complete label encoder using GAT for multilabel classification.
    Converts binary label vectors to structured embeddings.
    """
    def __init__(self, num_labels: int, embedding_dim: int = 128,
                 hidden_dims: list = [256, 128], num_heads: list = [4, 4],
                 pooling: str = 'cls', dropout: float = 0.1):
        super().__init__()
        self.num_labels = num_labels
        self.pooling = pooling
        
        # Validate pooling method
        valid_pooling = ['mean', 'max', 'sum', 'cls']
        if pooling not in valid_pooling:
            raise ValueError(f"pooling must be one of {valid_pooling}, got {pooling}")
        
        # Label embedding layer
        self.label_embedding = nn.Embedding(num_labels, embedding_dim)
        
        # Position encoding for label order
        # self.position_encoding = nn.Parameter(torch.randn(1, num_labels, embedding_dim))
        
        # CLS token for pooling='cls'
        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
            self.use_cls_token = True
        else:
            self.use_cls_token = False
        
        # GAT encoder
        self.gat_encoder = LearnableAdjacencyGAT(
            base_num_labels=num_labels,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            num_heads=num_heads,
            use_cls_token=self.use_cls_token
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output dimension
        self.output_dim = hidden_dims[-1]
    
    def forward(self, label_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            label_vector: [batch_size, num_labels] - Binary label vector
            
        Returns:
            [batch_size, output_dim] - Encoded label representation
            torch.Tensor: Learned adjacency matrix for analysis
        """
        batch_size = label_vector.shape[0]
        
        # Get label indices (for embedding lookup)
        label_indices = torch.arange(self.num_labels, device=label_vector.device)
        label_indices = label_indices.unsqueeze(0).expand(batch_size, -1)
        
        # Convert binary vector to weighted embeddings
        embeddings = self.label_embedding(label_indices) * label_vector.unsqueeze(-1)
        
        # Add position encoding
        embeddings = embeddings # + self.position_encoding
        
        # Add CLS token if using CLS pooling
        if self.pooling == 'cls':
            # Expand CLS token to batch size
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            # Concatenate CLS token at the beginning
            embeddings = torch.cat([cls_tokens, embeddings], dim=1)
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        # Apply GAT encoding
        gat_encoded, adj_matrix = self.gat_encoder(embeddings)
        
        # Pool to get global representation
        if self.pooling == 'mean':
            # Mean pooling over label tokens (excluding CLS if present)
            if self.use_cls_token:
                # Use only label tokens (exclude CLS at position 0)
                label_tokens = gat_encoded[:, 1:, :]
                pooled = label_tokens.mean(dim=1)
            else:
                pooled = gat_encoded.mean(dim=1)
        elif self.pooling == 'max':
            # Max pooling over label tokens
            if self.use_cls_token:
                label_tokens = gat_encoded[:, 1:, :]
                pooled = label_tokens.max(dim=1).values
            else:
                pooled = gat_encoded.max(dim=1).values
        elif self.pooling == 'sum':
            # Sum pooling over label tokens
            if self.use_cls_token:
                label_tokens = gat_encoded[:, 1:, :]
                pooled = label_tokens.sum(dim=1)
            else:
                pooled = gat_encoded.sum(dim=1)
        elif self.pooling == 'cls':
            # Use the CLS token representation
            pooled = gat_encoded[:, 0, :]  # CLS token is at position 0
        
        return pooled, adj_matrix


# Example usage
if __name__ == "__main__":
    batch_size = 32
    num_labels = 20
    
    # Create random binary label vectors
    label_vector = torch.randint(0, 2, (batch_size, num_labels)).float()
    
    # Initialize GAT label encoder
    label_encoder = LabelEncoderGAT(
        num_labels=num_labels,
        embedding_dim=128,
        hidden_dims=[256, 128],
        num_heads=[4, 4]
    )
    
    # Forward pass
    encoded_labels, adj_matrix = label_encoder(label_vector)
    
    print(f"Input shape: {label_vector.shape}")
    print(f"Encoded labels shape: {encoded_labels.shape}")
    print(f"Adjacency matrix shape: {adj_matrix.shape}")
    print(f"Sparsity of adjacency: {(adj_matrix[0] == 0).float().mean():.3f}")