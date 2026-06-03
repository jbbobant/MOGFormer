import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F

from src.models.layers.Structural_attention_block import StructuralAttentionBlock

class GlobalGraphTransformer(nn.Module):
    """
    Inter-Gene Network: Discovers distant trans-regulatory mechanisms across all genes
    using FlashAttention and Graph Positional Encodings.
    """
    def __init__(self, d: int = 64, pe_dim: int = 16, num_heads: int = 8, 
                 num_layers: int = 4, dim_feedforward: int = 256,
                 max_dist: int = 5, attention_mode: str = "boosted",dropout: float = 0.1):
        """
        Args:
            d: Token dimension (default 64)
            pe_dim: Dimensionality of the pre-computed Graph Positional Encodings
            num_heads: Number of attention heads
            num_layers: Number of transformer L layers
            dim_feedforward: Hidden dimension of the feed-forward network
            dropout: Attention and FFN dropout
        """
        super(GlobalGraphTransformer, self).__init__()
        self.d = d
        
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
        self.layers = nn.ModuleList([
            StructuralAttentionBlock(
                d_model=d, 
                num_heads=num_heads, 
                dim_feedforward=dim_feedforward,
                max_dist=max_dist,
                mode=attention_mode,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
    def forward(self, h: torch.Tensor, e_graph: torch.Tensor, spd_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Unified gene representations from Mini-Transformers. Shape (Batch, N_genes, d)
            e_graph: Pre-computed Graph PEs. Shape (N_genes, pe_dim)
            
        Returns:
            tumor_state: The final state of the [TUMOR_CLS] token. Shape (Batch, d)
            H_final: The full sequence output for potential downstream tasks. Shape (Batch, N_genes + 1, d)
        """
        B, N, D = h.shape

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

        padded_spd = F.pad(spd_matrix, (1, 0, 1, 0), value=0)

        H_curr = sequence
        for layer in self.layers:
            H_curr = layer(H_curr, padded_spd)
            
        return H_curr[:, 0, :], H_curr