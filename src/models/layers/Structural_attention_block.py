import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class StructuralAttentionBlock(nn.Module):
    """Combines StructuralGraphAttention with LayerNorms and FFN (Pre-LN style)"""
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, 
                 max_dist: int, mode: str, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = StructuralGraphAttention(
            d_model=d_model, num_heads=num_heads, max_dist=max_dist, mode=mode, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, spd_matrix: torch.Tensor) -> torch.Tensor:
        # Pre-LN architecture
        x = x + self.attn(self.norm1(x), spd_matrix)
        x = x + self.ffn(self.norm2(x))
        return x

class StructuralGraphAttention(nn.Module):
    """
    Multi-Head Attention supporting both Standard Graphormer 
    and Graphormer Boosted spatial bias injections.
    """
    def __init__(self, d_model: int, num_heads: int, max_dist: int = 5, mode: str = 'boosted', dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.mode = mode  # 'standard' or 'boosted'
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Spatial Bias Embedding: Maps discrete distances to scalars
        # Size is max_dist + 2 (to account for 0 distance and the disconnected threshold D_max + 1)
        self.spatial_embedding = nn.Embedding(max_dist + 2, num_heads)
        
        self.attn_drop = nn.Dropout(dropout)
        
    def forward(self, h: torch.Tensor, spd_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Node embeddings of shape (Batch, Seq_Len, d_model)
            spd_matrix: Discrete shortest path distances of shape (Seq_Len, Seq_Len)
        """
        B, N, _ = h.shape
        
        # 1. Project and reshape Q, K, V
        q = self.q_proj(h).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(h).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(h).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        
        # 2. Compute semantic similarity (S)
        s_matrix = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        # 3. Compute structural spatial bias (B)
        # Shape becomes (Seq_Len, Seq_Len, Num_Heads) -> (Num_Heads, Seq_Len, Seq_Len)
        spatial_bias = self.spatial_embedding(spd_matrix).permute(2, 0, 1)
        # Expand spatial bias across the batch dimension
        spatial_bias = spatial_bias.unsqueeze(0).expand(B, -1, -1, -1)
        
        # Apply tanh bounding as specified in the MOGFormer design
        spatial_bias = torch.tanh(spatial_bias)
        
        # 4. Attention Score Unification
        if self.mode == 'standard':
            # Original Graphormer: A = softmax(S + B)
            raw_attn = s_matrix + spatial_bias
            attn_weights = F.softmax(raw_attn, dim=-1)
            
        elif self.mode == 'boosted':
            # Graphormer Boosted: A = softmax(S) + softmax(B) [cite: 425, 431]
            s_norm = F.softmax(s_matrix, dim=-1)
            b_norm = F.softmax(spatial_bias, dim=-1)
            # Re-normalize to ensure rows sum to 1
            attn_weights = (s_norm + b_norm) / 2.0
            
        else:
            raise ValueError("Mode must be either 'standard' or 'boosted'.")
            
        attn_weights = self.attn_drop(attn_weights)
        
        # 5. Apply to values and project
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        
        return self.out_proj(out)