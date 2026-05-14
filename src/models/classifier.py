import torch
import torch.nn as nn
from typing import Dict, Tuple

# Assuming these are imported from your local project structure
from .layers.modality_lifting import ModalityLifting
from .layers.mini_transformer import MiniTransformer
from .layers.global_transformer import GlobalGraphTransformer

class MultiOmicsGraphClassifier(nn.Module):
    """
    The full Patient-Specific Multi-Omics Graph Transformer.
    Integrates mRNA, CNV, and Methylation data with PPI structural priors
    to predict breast cancer subtypes.
    """
    def __init__(self, 
                 num_classes: int = 5, 
                 d: int = 64, 
                 pe_dim: int = 16,
                 mini_heads: int = 4,
                 global_heads: int = 8,
                 global_layers: int = 4,
                 dropout: float = 0.1,
                 rna_dropout_prob: float = 0.15):
        """
        Args:
            num_classes: Number of clinical subtypes (e.g., 5 for BRCA)
            d: Unified token dimension
            pe_dim: Dimensionality of Graph Positional Encodings
        """
        super(MultiOmicsGraphClassifier, self).__init__()
        
        self.d = d
        self.num_classes = num_classes
        
        # 1. Modality Lifting
        self.lifter = ModalityLifting(d=d)
        
        # 2. Intra-Gene Fusion (Mini-Transformer)
        self.mini_transformer = MiniTransformer(
            d=d, 
            num_heads=mini_heads, 
            dropout=dropout, 
            rna_dropout_prob=rna_dropout_prob
        )
        
        # 3. Inter-Gene Network (Global Graph Transformer)
        self.global_transformer = GlobalGraphTransformer(
            d=d, 
            pe_dim=pe_dim, 
            num_heads=global_heads, 
            num_layers=global_layers, 
            dropout=dropout
        )
        
        # 4. Classification Head (MLP)
        # Passes the [TUMOR_CLS] token through an MLP to predict subtypes.
        self.classifier_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.BatchNorm1d(d // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d // 2, num_classes)
            # Note: Softmax is omitted here for nn.CrossEntropyLoss stability.
        )

    def forward(self, 
                rna: torch.Tensor, 
                cnv: torch.Tensor, 
                methy: torch.Tensor, 
                e_graph: torch.Tensor,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            rna, cnv, methy: Modality tensors of shape (Batch, N_genes)
            e_graph: Graph PEs of shape (N_genes, pe_dim)
            return_attention: If True, returns interpretability data
            
        Returns:
            Dictionary containing logits and optionally attention weights/embeddings.
        """
        # Step 1: Modality Lifting -> Project to dimension d and add biological origin embeddings
        z_m, z_c, z_t = self.lifter(rna, cnv, methy)
        
        # Step 2: Mini-Transformer -> Intra-gene cross-attention to output unified gene representations
        h, intra_attn_weights = self.mini_transformer(z_m, z_c, z_t)
        
        # Step 3: Global Transformer -> Structural injection and [TUMOR_CLS] aggregation
        tumor_state, H_final = self.global_transformer(h, e_graph)
        
        # Step 4: Subtype Prediction -> MLP on the final [TUMOR_CLS] state 
        logits = self.classifier_head(tumor_state)
        
        # Prepare output dictionary
        output = {"logits": logits}
        
        # Native Interpretability Strategy support
        if return_attention:
            output["tumor_state"] = tumor_state
            output["intra_attn_weights"] = intra_attn_weights
            # We can expand this later to include the top K driver genes extraction[cite: 42].
            
        return output