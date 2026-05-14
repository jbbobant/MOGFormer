import torch
import torch.nn as nn
from typing import Tuple, Dict

class ModalityLifting(nn.Module):
    """
    Projects raw scalar multi-omics features into a d-dimensional continuous space
    and injects learnable biological modality embeddings.
    """
    def __init__(self, d: int = 64):
        super(ModalityLifting, self).__init__()
        self.d = d
        
        # 1. Modality-Specific Linear Projections
        # Input is 1 scalar feature per gene, projected to d dimensions.
        self.linear_rna = nn.Linear(1, d)
        self.linear_cnv = nn.Linear(1, d)
        self.linear_methy = nn.Linear(1, d)
        
        # 2. Modality Embeddings
        # Learnable vectors added to signify biological origin (mRNA, CNV, Methylation)
        # Shape: (1, 1, d) to allow broadcasting across Batch and N_genes
        self.emb_rna = nn.Parameter(torch.randn(1, 1, d))
        self.emb_cnv = nn.Parameter(torch.randn(1, 1, d))
        self.emb_methy = nn.Parameter(torch.randn(1, 1, d))
        
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using Xavier Normal for stable early training."""
        nn.init.xavier_normal_(self.linear_rna.weight)
        nn.init.xavier_normal_(self.linear_cnv.weight)
        nn.init.xavier_normal_(self.linear_methy.weight)
        
        # Initialize biases to zero
        nn.init.zeros_(self.linear_rna.bias)
        nn.init.zeros_(self.linear_cnv.bias)
        nn.init.zeros_(self.linear_methy.bias)
        
        # Initialize embeddings with small random values
        nn.init.normal_(self.emb_rna, mean=0.0, std=0.02)
        nn.init.normal_(self.emb_cnv, mean=0.0, std=0.02)
        nn.init.normal_(self.emb_methy, mean=0.0, std=0.02)

    def forward(self, rna: torch.Tensor, cnv: torch.Tensor, methy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            rna: Tensor of shape (Batch, N_genes)
            cnv: Tensor of shape (Batch, N_genes)
            methy: Tensor of shape (Batch, N_genes)
            
        Returns:
            Tuple of projected and embedded tensors: z_m, z_c, z_t
            Each output has shape (Batch, N_genes, d)
        """
        # Expand inputs from (Batch, N_genes) to (Batch, N_genes, 1) for the Linear layer
        rna_expanded = rna.unsqueeze(-1)
        cnv_expanded = cnv.unsqueeze(-1)
        methy_expanded = methy.unsqueeze(-1)
        
        # Apply Linear Projections: z_m = Linear_mRNA(x_mRNA)
        z_m = self.linear_rna(rna_expanded)
        z_c = self.linear_cnv(cnv_expanded)
        z_t = self.linear_methy(methy_expanded)
        
        # Add Biological Modality Embeddings (Broadcasting handles the Batch and N_genes dims)
        z_m = z_m + self.emb_rna
        z_c = z_c + self.emb_cnv
        z_t = z_t + self.emb_methy
        
        return z_m, z_c, z_t