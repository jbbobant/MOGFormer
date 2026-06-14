import torch
import torch.nn as nn
from typing import Tuple, Dict

class GeneEmbeddingAdapter(nn.Module):
    """Frozen pre-trained gene embedding + optional trainable bottleneck adapter.

    adapter_rank=0 : linear projection W_G only (frozen base, trained projection)
    adapter_rank>0 : LoRA-style residual correction before projection
    """
    def __init__(self, pretrained: torch.Tensor, d_out: int, adapter_rank: int = 0):
        super().__init__()
        d_e = pretrained.shape[1]
        self.register_buffer("base", pretrained.float())   # frozen, moves with .to(device)
        self.adapter_rank = adapter_rank
        if adapter_rank > 0:
            self.W_down = nn.Linear(d_e, adapter_rank, bias=False)
            self.W_up   = nn.Linear(adapter_rank, d_e, bias=False)
            nn.init.xavier_normal_(self.W_down.weight)
            nn.init.zeros_(self.W_up.weight)   # zero init -> identity residual at t=0
        self.proj = nn.Linear(d_e, d_out, bias=False)
        nn.init.xavier_normal_(self.proj.weight)

    def forward(self) -> torch.Tensor:
        # base is already sliced to selected genes (n_sel, d_e)
        x = self.base
        if self.adapter_rank > 0:
            x = x + self.W_up(torch.nn.functional.gelu(self.W_down(x)))
        return self.proj(x).unsqueeze(0)   # (1, n_sel, d) — broadcasts over batch

class PLRModality(nn.Module):
    """Periodic–Linear–ReLU numerical tokenizer (Gorishniy et al., NeurIPS 2022).
 
    Maps a scalar value x to a d-dimensional token:
        periodic(x) = [sin(2π v_k x), cos(2π v_k x)]_{k=1..K}   v_k learnable ~ N(0,σ²)
        out = Linear_2( ReLU( Linear_1( periodic(x) ) ) )          P → L → R → L
 
    One instance per modality (RNA / CNV / methy); frequencies are per-modality and
    shared across genes. Gene identity is carried separately by G_i (Change A), so
    the tokenizer learns purely the modality-level value distribution.
 
    σ (sigma) is the most sensitive hyperparameter: it controls the frequency
    spectrum of the periodic basis. For StandardScaled features, σ ≈ 1.0 is a
    reasonable start; tune over {0.1, 0.5, 1.0, 3.0} in the inner validation loop.
    """
 
    def __init__(self, d: int, K: int = 16, sigma: float = 1.0):
        super().__init__()
        self.K = K
        # Learnable frequencies v_k; init ~ N(0, σ²) as in the original paper.
        self.frequencies = nn.Parameter(torch.randn(K) * sigma)
        self.linear1 = nn.Linear(2 * K, d)
        self.linear2 = nn.Linear(d, d)
        nn.init.xavier_normal_(self.linear1.weight); nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_normal_(self.linear2.weight); nn.init.zeros_(self.linear2.bias)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N)  ->  out: (B, N, d)
        x_proj = x.unsqueeze(-1) * self.frequencies * 2.0 * math.pi   # (B, N, K)
        periodic = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # (B, N, 2K)
        return self.linear2(torch.relu(self.linear1(periodic)))         # (B, N, d)
 
 
class ModalityLifting(nn.Module):
    """Per-gene multimodal tokenizer.
 
    Supports two numerical tokenizers (controlled by numerical_tokenizer):
      "mlp" : two-layer MLP with LayerNorm (original)
      "plr" : PLRModality — Periodic-Linear-ReLU (Change C)
 
    Change A: optional per-gene identity embedding G_i (keyed by universe position)
    added to all three modality tokens after tokenization.
    """
 
    def __init__(self, d: int = 64,
                 gene_id_embedding: bool = False,
                 n_universe: int = 0,
                 gene_ids: Optional[Sequence[int]] = None,
                 gene_id_std: float = 0.02,
                 numerical_tokenizer: str = "mlp",
                 plr_n_frequencies: int = 16,
                 plr_sigma: float = 1.0,
                 unimodal_dropout_fill: str = "zero",
                 pretrained_gene_emb=None,            # (n_sel, d_e) tensor or None
                 pretrained_emb_adapter_rank=0        # 0 -> projection only 
                 ):
        super().__init__()
        self.d = d
        self.gene_id_embedding = gene_id_embedding
        self.numerical_tokenizer = numerical_tokenizer
 
        if numerical_tokenizer == "plr":
            self.tok_rna   = PLRModality(d, K=plr_n_frequencies, sigma=plr_sigma)
            self.tok_cnv   = PLRModality(d, K=plr_n_frequencies, sigma=plr_sigma)
            self.tok_methy = PLRModality(d, K=plr_n_frequencies, sigma=plr_sigma)
        else:                                              # "mlp" — original behaviour
            self.tok_rna = nn.Sequential(
                nn.Linear(1, d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d))
            self.tok_cnv = nn.Sequential(
                nn.Linear(1, d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d))
            self.tok_methy = nn.Sequential(
                nn.Linear(1, d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d))
 
        self.emb_rna   = nn.Parameter(torch.randn(1, 1, d))
        self.emb_cnv   = nn.Parameter(torch.randn(1, 1, d))
        self.emb_methy = nn.Parameter(torch.randn(1, 1, d))
 
        if self.gene_id_embedding:
            if pretrained_gene_emb is not None:
                # Pre-trained path: adapter replaces the nn.Embedding
                self.gene_adapter = GeneEmbeddingAdapter(
                    pretrained=pretrained_gene_emb,
                    d_out=d,
                    adapter_rank=pretrained_emb_adapter_rank)
            else:
                # Learnable path: original Change A
                if not (n_universe and gene_ids is not None):
                    raise ValueError("gene_id_embedding=True needs n_universe>0 and gene_ids.")
                self.gene_emb = nn.Embedding(n_universe, d)
                self.register_buffer("gene_ids", torch.as_tensor(gene_ids, dtype=torch.long))
                self._gene_id_std = gene_id_std
 
        self._init_weights()
 
    def _init_weights(self):
        if self.numerical_tokenizer == "mlp":
            for layer in [self.tok_rna, self.tok_cnv, self.tok_methy]:
                nn.init.xavier_normal_(layer[0].weight); nn.init.zeros_(layer[0].bias)
                nn.init.xavier_normal_(layer[3].weight); nn.init.zeros_(layer[3].bias)
        # PLR linears are initialised inside PLRModality.__init__
        nn.init.normal_(self.emb_rna,   mean=0.0, std=0.02)
        nn.init.normal_(self.emb_cnv,   mean=0.0, std=0.02)
        nn.init.normal_(self.emb_methy, mean=0.0, std=0.02)
        
        if self.gene_id_embedding and hasattr(self, 'gene_emb'):
            nn.init.normal_(self.gene_emb.weight, mean=0.0, std=self._gene_id_std)
 
    def forward(self, rna: torch.Tensor, cnv: torch.Tensor, methy: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.numerical_tokenizer == "plr":
            z_m = self.tok_rna(rna)   + self.emb_rna     # (B, N, d)
            z_c = self.tok_cnv(cnv)   + self.emb_cnv
            z_t = self.tok_methy(methy) + self.emb_methy
        else:
            z_m = self.tok_rna(rna.unsqueeze(-1))   + self.emb_rna
            z_c = self.tok_cnv(cnv.unsqueeze(-1))   + self.emb_cnv
            z_t = self.tok_methy(methy.unsqueeze(-1)) + self.emb_methy
 
        if self.gene_id_embedding:
            if hasattr(self, 'gene_adapter'):
                G = self.gene_adapter()              # (1, n_sel, d) from pre-trained
            else:
                G = self.gene_emb(self.gene_ids).unsqueeze(0)   # (1, n_sel, d) learnable
            z_m, z_c, z_t = z_m + G, z_c + G, z_t + G
 
        return z_m, z_c, z_t