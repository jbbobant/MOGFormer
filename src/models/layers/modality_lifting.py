import torch
import torch.nn as nn
from typing import Tuple, Dict

class ModalityLifting(nn.Module):
    def __init__(self, d: int = 64):
        super(ModalityLifting, self).__init__()
        self.d = d
        
        # Upgraded to Two-Layer MLPs with LayerNorm and non-linearities
        self.linear_rna = nn.Sequential(
            nn.Linear(1, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Linear(d, d)
        )
        self.linear_cnv = nn.Sequential(
            nn.Linear(1, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Linear(d, d)
        )
        self.linear_methy = nn.Sequential(
            nn.Linear(1, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Linear(d, d)
        )

        self.emb_rna = nn.Parameter(torch.randn(1, 1, d))
        self.emb_cnv = nn.Parameter(torch.randn(1, 1, d))
        self.emb_methy = nn.Parameter(torch.randn(1, 1, d))
        self._init_weights()

    def _init_weights(self):
        # Apply initialization to the internal linear layers
        for layer in [self.linear_rna, self.linear_cnv, self.linear_methy]:
            nn.init.xavier_normal_(layer[0].weight)
            nn.init.zeros_(layer[0].bias)
            nn.init.xavier_normal_(layer[3].weight)
            nn.init.zeros_(layer[3].bias)
        
        nn.init.normal_(self.emb_rna, mean=0.0, std=0.02)
        nn.init.normal_(self.emb_cnv, mean=0.0, std=0.02)
        nn.init.normal_(self.emb_methy, mean=0.0, std=0.02)

    def forward(self, rna: torch.Tensor, cnv: torch.Tensor, methy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_m = self.linear_rna(rna.unsqueeze(-1)) + self.emb_rna
        z_c = self.linear_cnv(cnv.unsqueeze(-1)) + self.emb_cnv
        z_t = self.linear_methy(methy.unsqueeze(-1)) + self.emb_methy
        return z_m, z_c, z_t