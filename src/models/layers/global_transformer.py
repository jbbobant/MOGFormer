import torch
import torch.nn as nn
from typing import Tuple

class GlobalGraphTransformer(nn.Module):
    """
    Inter-Gene Network: Discovers distant trans-regulatory mechanisms across all genes
    using FlashAttention and Graph Positional Encodings.
    """
    def __init__(self,base_adj: torch.Tensor, d: int = 64, pe_dim: int = 16, num_heads: int = 8, 
                 num_layers: int = 4, dim_feedforward: int = 256, dropout: float = 0.1 ):
        """
        Args:
            d: Token dimension (default 64)
            pe_dim: Dimensionality of the pre-computed Graph Positional Encodings
            num_heads: Number of attention heads
            num_layers: Number of transformer L layers
            dim_feedforward: Hidden dimension of the feed-forward network
            dropout: Attention and FFN dropout
            base_adj: Adjacency Matrix of GRN, PPI...
        """
        super(GlobalGraphTransformer, self).__init__()
        self.d = d
        self.register_buffer('base_adj', base_adj.float())
        
        # 1. Structural Injection (Graph PE Projector)
        # Projector for the concatenated features (from pe_dim + d to d)

        self.concat_projector = nn.Linear(d + pe_dim, d)
        
        # 2. The Global Sponge: Master [TUMOR_CLS] Token
        # Shape: (1, 1, d) to broadcast across the batch size

        self.tumor_cls = nn.Parameter(torch.randn(1, 1, d))
        nn.init.normal_(self.tumor_cls, mean=0.0, std=0.02)
        
        # 3. Linear Self-Attention Layers (FlashAttention)
        # In PyTorch 2.0+, batch_first=True natively triggers FlashAttention backend
        # for highly efficient memory usage, satisfying the O(N) requirement.

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True # Pre-LN architecture
        )

        self.transformer_layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.graph_bias_scalar = nn.Parameter(torch.tensor([0.5]))

    def forward(self, h: torch.Tensor, e_graph: torch.Tensor, base_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Unified gene representations from Mini-Transformers. Shape (Batch, N_genes, d)
            e_graph: Pre-computed Graph PEs. Shape (N_genes, pe_dim)
            
        Returns:
            tumor_state: The final state of the [TUMOR_CLS] token. Shape (Batch, d)
            H_final: The full sequence output for potential downstream tasks. Shape (Batch, N_genes + 1, d)
        """
        structural_bias = (self.base_adj - 1.0) * 2.0  # (N, N)

        B, N, D = h.shape

        full_bias = torch.zeros(N + 1, N + 1, device=h.device)
        
        full_bias[1:, 1:] = structural_bias * self.graph_bias_scalar

        # -- STRUCTURAL INJECTION (Concatenation & Projection) --
        # 1. Expand e_graph to match the batch size: (N_genes, pe_dim) -> (Batch, N_genes, pe_dim)
        e_graph_expanded = e_graph.unsqueeze(0).expand(B, -1, -1)
        
        # 2. Concatenate features and structural encodings along the feature dimension
        # New shape: (Batch, N_genes, d + pe_dim)
        h_concat = torch.cat([h, e_graph_expanded], dim=-1)
    
        # 3. Project the concatenated representation back to the token dimension 'd'
        # New shape: (Batch, N_genes, d)
        H_0 = self.concat_projector(h_concat)
    
        # -- THE GLOBAL SPONGE --
        tumor_cls_expanded = self.tumor_cls.expand(B, 1, D)
        sequence = torch.cat([tumor_cls_expanded, H_0], dim=1)
    
        H_final = self.transformer_layers(sequence, mask = full_bias)
    
        # Extract the final state of the [TUMOR_CLS] token (Index 0)
        tumor_state = H_final[:, 0, :]
        
        return tumor_state, H_final