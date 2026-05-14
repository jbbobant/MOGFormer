import torch
import torch.nn as nn
from typing import Tuple

class GlobalGraphTransformer(nn.Module):
    """
    Inter-Gene Network: Discovers distant trans-regulatory mechanisms across all genes
    using FlashAttention and Graph Positional Encodings.
    """
    def __init__(self, d: int = 64, pe_dim: int = 16, num_heads: int = 8, 
                 num_layers: int = 4, dim_feedforward: int = 256, dropout: float = 0.1):
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
        # Projects the pe_dim (e.g., 16) up to token dimension d (e.g., 64)
        self.pe_projector = nn.Linear(pe_dim, d)
        
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

    def forward(self, h: torch.Tensor, e_graph: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Unified gene representations from Mini-Transformers. Shape (Batch, N_genes, d)
            e_graph: Pre-computed Graph PEs. Shape (N_genes, pe_dim)
            
        Returns:
            tumor_state: The final state of the [TUMOR_CLS] token. Shape (Batch, d)
            H_final: The full sequence output for potential downstream tasks. Shape (Batch, N_genes + 1, d)
        """
        B, N, D = h.shape
        
        # -- STRUCTURAL INJECTION --
        # Project PE to token dimension: (N_genes, pe_dim) -> (N_genes, d)
        e_graph_projected = self.pe_projector(e_graph)
        
        # Add PE directly to the gene tokens: H^(0) = [h_1, ..., h_N] + E_graph
        # e_graph_projected broadcasts across the Batch dimension automatically
        H_0 = h + e_graph_projected
        
        # -- THE GLOBAL SPONGE --
        # Expand the master [TUMOR_CLS] token to match the batch size: (Batch, 1, d)
        tumor_cls_expanded = self.tumor_cls.expand(B, 1, D)
        
        # Prepend the token to the entire sequence: New shape (Batch, N_genes + 1, d)
        # Index 0 is now the [TUMOR_CLS] token
        sequence = torch.cat([tumor_cls_expanded, H_0], dim=1)
        
        # -- GLOBAL ATTENTION --
        # Pass through the L transformer layers
        # The [TUMOR_CLS] token attends to all N gene tokens, aggregating systemic signals
        H_final = self.transformer_layers(sequence)
        
        # Extract the final state of the [TUMOR_CLS] token (Index 0)
        tumor_state = H_final[:, 0, :]
        
        return tumor_state, H_final