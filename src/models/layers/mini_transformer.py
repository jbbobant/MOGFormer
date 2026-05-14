import torch
import torch.nn as nn
from typing import Tuple

class MiniTransformer(nn.Module):
    """
    Intra-Gene Attention mechanism. Fuses mRNA, CNV, and Methylation data 
    into a single unified gene representation via self-attention.
    """
    def __init__(self, d: int = 64, num_heads: int = 4, dropout: float = 0.1, rna_dropout_prob: float = 0):
        """
        Args:
            d: Token dimension (default 64)
            num_heads: Number of attention heads for the multi-head attention
            dropout: Standard attention dropout
            rna_dropout_prob: Probability of masking the mRNA modality during training (10-20%)
        """
        super(MiniTransformer, self).__init__()
        self.d = d
        self.rna_dropout_prob = rna_dropout_prob
        
        # 1. Learnable Gene-Level Classification Token (z_cls)
        # Shape: (1, 1, 1, d) so it can broadcast across Batch and N_genes
        self.z_cls = nn.Parameter(torch.randn(1, 1, 1, d))
        nn.init.normal_(self.z_cls, mean=0.0, std=0.02)
        
        # 2. Scaled Dot-Product Attention (using PyTorch's MultiheadAttention for efficiency)
        # batch_first=True means input shape is (Batch, SequenceLength, Features)
        self.attention = nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, 
                                               dropout=dropout, batch_first=True)
        
        # Layer Normalization and Feed Forward (standard Transformer components)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d)
        )

    def forward(self, z_m: torch.Tensor, z_c: torch.Tensor, z_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_m, z_c, z_t: Modality tensors of shape (Batch, N_genes, d)
            
        Returns:
            h_i: The extracted context-aware gene representations (Batch, N_genes, d)
            attn_weights: Attention weights for interpretability (Batch * N_genes, Num_Heads, 4, 4)
        """
        B, N, D = z_m.shape
        
        # -- REGULARIZATION: Modality Dropout --
        # Randomly zero-out the mRNA embedding for a percentage of genes in a batch during training
        if self.training and self.rna_dropout_prob > 0.0:
            # Create a mask of shape (Batch, N_genes, 1)
            mask = (torch.rand(B, N, 1, device=z_m.device) > self.rna_dropout_prob).float()
            z_m = z_m * mask

        # -- SEQUENCE FORMULATION --
        # Stack modalities to shape: (Batch, N_genes, 3, d)
        modalities = torch.stack([z_m, z_c, z_t], dim=2)
        
        # Expand z_cls to match Batch and N_genes: (Batch, N_genes, 1, d)
        cls_expanded = self.z_cls.expand(B, N, 1, D)
        
        # Prepend z_cls to form X_i. New shape: (Batch, N_genes, 4, d)
        # Sequence indices: 0 -> z_cls, 1 -> z_m, 2 -> z_c, 3 -> z_t
        X_i = torch.cat([cls_expanded, modalities], dim=2)
        
        # -- INTRA-GENE ATTENTION --
        # PyTorch MHA expects a single batch dimension. We merge Batch and N_genes.
        # Shape becomes: (Batch * N_genes, 4, d)
        X_i_flat = X_i.view(B * N, 4, D)
        
        # Apply Layer Norm before attention (Pre-LN architecture is more stable)
        X_i_norm = self.norm1(X_i_flat)
        
        # Q, K, V are all derived from X_i_norm
        attn_out, attn_weights = self.attention(
            query=X_i_norm, 
            key=X_i_norm, 
            value=X_i_norm,
            need_weights=True
        )
        
        # Residual connection
        X_i_flat = X_i_flat + attn_out
        
        # FFN with residual connection
        X_i_flat = X_i_flat + self.ffn(self.norm2(X_i_flat))
        
        # -- TOKEN EXTRACTION --
        # Reshape back to (Batch, N_genes, 4, d)
        X_i_updated = X_i_flat.view(B, N, 4, D)
        
        # Extract the updated z_cls token (index 0 in the sequence dimension)
        # This is h_i: The unified biological representation of the gene
        h_i = X_i_updated[:, :, 0, :]
        
        return h_i, attn_weights