"""
evaluation/runner.py — consolidated K-fold trainer + orchestration.

Per fold the runner: fits MultiOmicsTransformer on outer-train -> selected genes ->
induced StRING subgraph A -> PE + SPD computed ONCE from A -> FoldTrainer : Training and evaluation fold 

"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, is_dataclass
from typing import Optional

import numpy as np


import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import pandas as pd
import yaml

from src.evaluation import fold_plots
from src.evaluation.cv import  load_folds
from src.evaluation.fold_data import load_omics, load_curated_genes
from src.evaluation.fold_metrics import compute_fold_metrics, fold_confusion, aggregate
from src.evaluation.fold_preprocess import MultiOmicsTransformer
from src.graph.string_graph import StringGraphCache
from src.graph.positional_encoding import GraphPositionalEncoding
from src.graph.spd import compute_shortest_path_matrix
from src.utils.loss import MultiClassFocalLoss
from src.models.classifier import MultiOmicsGraphClassifier


# --------------------------------------------------------------------------
@dataclass
class MOGFormerConfig:
    # ---- locked BASELINE (script.py) ----
    d: int = 128                        # embbeding dimension
    pe_dim: int = 32                    # positional encoding dimension
    pe_method: str = "rwpe"             # "rwpe" | "laplacian" 
    mini_heads: int = 4                 # number of mini-transformer heads
    global_heads: int = 8               # number of global-transformer heads
    global_layers: int = 1              # number of global-transformer layers
    dropout: float = 0.2                # global dropout
    rna_dropout: float = 0.4            # rna-seq dropout
    cnv_dropout: float = 0.2            # cnv dropout
    meth_dropout: float = 0.2           # methy dropout
    max_dist: int = 10                  # maximum distance between genes in the graph
    attention_mode: str = "boosted"     # attention mode : "boosted" | "standard"
    lr: float = 1e-4                    # learning rate start
    min_lr: float = 1e-6                # learning rate end
    weight_decay: float = 1e-4          # weight decay
    max_epochs: int = 400               # maximum number of epochs
    patience: int = 40                  # patience for early stopping
    log_every: int = 1                  # log every n epochs
    batch_size: int = 32                # batch size
    seed: int = 42                      # random seed
    top_k: int = 250                    # top k genes to select
    string_threshold: int = 400         # confidence threshold for PPI
    inner_val_frac: float = 0.20        # fraction of training data to use for validation
    focal_gamma: float = 2.0            # gamma parameter for focal loss
    # ---- fold control ----
    n_repeats_used: int = 5             # 5 -> full 25 estimates; 1 -> single repeat (5 folds)
    max_folds: Optional[int] = None     # None -> all; 1 -> single fold 
    # ---- Phase-1 flags (default == current behavior; model untouched) ----
    gene_id_embedding: bool = False     # enable gene ID embedding
    attention_bias_mode: str = "dual_softmax"  # "dual_softmax" | "standard"
    numerical_tokenizer: str = "mlp"            # "mlp" | "linear"
    unimodal_dropout_fill: str = "zero"       # "zero" | "mean"


# ---- torch-free helpers --------------------------------------------------
def split_modalities(Xt: np.ndarray, n_sel: int):
    return Xt[:, :n_sel], Xt[:, n_sel:2 * n_sel], Xt[:, 2 * n_sel:3 * n_sel]


def sqrt_dampened_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(float)
    counts = np.clip(counts, 1.0, None)
    damp = np.sqrt(1.0 / counts)
    return damp / damp.sum() * num_classes


def assert_folds_match(loaded_patient_order, current_patient_ids):
    if list(loaded_patient_order) != list(current_patient_ids):
        raise AssertionError(
            "folds.json patient_order != current load_omics order; MOGFormer would "
            "not be paired with the Phase 0 baselines. Reuse the exact Phase 0 folds.")


def select_folds(folds, n_repeats_used, max_folds):
    folds = [f for f in folds if f.repeat < n_repeats_used]
    if max_folds is not None:
        folds = folds[:max_folds]
    return folds


def _config_to_dict(config):
    if is_dataclass(config):
        return asdict(config)
    return {k: v for k, v in vars(config).items() if not k.startswith("_")}


# ---- merged trainer ------------------------------------------------------
class FoldTrainer:
    """Self-contained per-fold trainer. PE + SPD + criterion are built once per
    fold by the runner and passed in (they are fold-constants)."""

    def __init__(self, model, train_loader, val_loader, device, spd_matrix,
                 graph_pe, criterion, lr, min_lr, weight_decay):
        import torch.optim as optim
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.spd_matrix = spd_matrix
        self.graph_pe = graph_pe
        self.criterion = criterion
        self.min_lr = min_lr
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.best_val_f1 = 0.0
        self.history = {"train_loss": [], "val_loss": [], "val_f1": []}

    def _train_epoch(self) -> float:
        import torch
        from tqdm import tqdm
        self.model.train()
        total = 0.0
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            rna = batch["mRNA"].to(self.device)
            cnv = batch["CNV"].to(self.device)
            methy = batch["methy"].to(self.device)
            labels = batch["label"].to(self.device)
            self.optimizer.zero_grad()
            out = self.model(rna, cnv, methy, self.graph_pe, self.spd_matrix)
            loss = self.criterion(out["logits"], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        return total / len(self.train_loader)

    def _validate_epoch(self) -> dict:
        import torch
        from sklearn.metrics import f1_score
        self.model.eval()
        total, preds, labs = 0.0, [], []
        with torch.no_grad():
            for batch in self.val_loader:
                rna = batch["mRNA"].to(self.device)
                cnv = batch["CNV"].to(self.device)
                methy = batch["methy"].to(self.device)
                labels = batch["label"].to(self.device)
                out = self.model(rna, cnv, methy, self.graph_pe, self.spd_matrix)
                total += self.criterion(out["logits"], labels).item()
                preds.extend(torch.argmax(out["logits"], 1).cpu().numpy())
                labs.extend(labels.cpu().numpy())
        return {"val_loss": total / len(self.val_loader),
                "val_f1": f1_score(labs, preds, average="macro")}

    def fit_early(self, max_epochs: int, patience: int, log_every: int = 1) -> float:
        import torch.optim as optim
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs, eta_min=self.min_lr)
        best_f1, best_state, wait, best_epoch = -1.0, None, 0, 0
        best_tl, best_vl = float("nan"), float("nan")
        epoch = 0
        for epoch in range(1, max_epochs + 1):
            lr = self.optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:>3}/{max_epochs}  (patience {wait}/{patience}, lr {lr:.2e})")
            tl = self._train_epoch()
            vm = self._validate_epoch()
            self.scheduler.step()
            vl, vf = vm["val_loss"], vm["val_f1"]
            self.history["train_loss"].append(tl)
            self.history["val_loss"].append(vl)
            self.history["val_f1"].append(vf)
            improved = vf > best_f1
            if improved:
                best_f1, best_epoch, best_tl, best_vl = vf, epoch, tl, vl
                best_state = copy.deepcopy(self.model.state_dict())
                wait = 0
            else:
                wait += 1
            if improved or epoch % max(1, log_every) == 0 or wait >= patience:
                flag = "  *new best*" if improved else ""
                print(f"  -> train {tl:.4f} | val {vl:.4f} | val_F1 {vf:.4f} | "
                      f"best {best_f1:.4f}@e{best_epoch}{flag}")
            if wait >= patience:
                print(f"  early-stop: no val_F1 gain for {patience} epochs.")
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_val_f1 = best_f1
        self.best_epoch = best_epoch
        self.best_train_loss = best_tl
        self.best_val_loss = best_vl
        self.stopped_epoch = epoch
        return best_f1

    def predict_proba(self, loader):
        import torch
        import torch.nn.functional as F
        self.model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for batch in loader:
                rna = batch["mRNA"].to(self.device)
                cnv = batch["CNV"].to(self.device)
                methy = batch["methy"].to(self.device)
                out = self.model(rna, cnv, methy, self.graph_pe, self.spd_matrix)
                probs.append(F.softmax(out["logits"], dim=1).cpu().numpy())
                labels.append(batch["label"].numpy())
        return np.vstack(probs), np.concatenate(labels)


def _make_tensor_dataset():
    import torch
    from torch.utils.data import Dataset

    class OmicsTensorDataset(Dataset):
        def __init__(self, rna, cnv, methy, y):
            self.rna, self.cnv, self.methy, self.y = rna, cnv, methy, y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return {"mRNA": torch.tensor(self.rna[i], dtype=torch.float32),
                    "CNV": torch.tensor(self.cnv[i], dtype=torch.float32),
                    "methy": torch.tensor(self.methy[i], dtype=torch.float32),
                    "label": torch.tensor(int(self.y[i]), dtype=torch.long)}
    return OmicsTensorDataset


# ---- orchestration -------------------------------------------------------
def run_mogformer_cv(config: MOGFormerConfig, data_dir: str, results_dir: str,
                     folds_path: str,
                     ppi_file: str = "9606.protein.links.v12.0.txt",
                     alias_file: str = "9606.protein.aliases.v12.0.txt",
                     curated_genes_file: str = None,
                     files: dict = None):

    files = files or {"rna": "data_rna_seq_v2_rsem.csv", "cnv": "data_cnv.csv",
                      "methy": "data_methylation450.csv", "clin": "data_clinical.csv"}
    os.makedirs(results_dir, exist_ok=True)
    fig_dir = os.path.join(results_dir, "figures"); os.makedirs(fig_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # STEP 0 guard: refuse silently-inert flipped Phase-1 flags.
    _CURRENT = {"gene_id_embedding": False, "attention_bias_mode": "dual_softmax",
                "numerical_tokenizer": "mlp", "unimodal_dropout_fill": "zero"}
    flipped = {k: getattr(config, k) for k, v in _CURRENT.items() if getattr(config, k) != v}
    if flipped:
        raise NotImplementedError(f"Phase-1 flag(s) flipped but not yet wired: {flipped}.")

    DS = _make_tensor_dataset()
    od = load_omics(data_dir, files["rna"], files["cnv"], files["methy"], files["clin"])
    curated = load_curated_genes(curated_genes_file, od.gene_names)
    folds, patient_order = load_folds(folds_path)
    assert_folds_match(patient_order, od.patient_ids)
    folds = select_folds(folds, config.n_repeats_used, config.max_folds)
    print(f"[folds] using {len(folds)} estimate(s)")
    num_classes = len(od.label_map)
    class_names = [od.inverse_label_map[i] for i in range(num_classes)]
    labels_idx = list(range(num_classes))

    cache = StringGraphCache(os.path.join(data_dir, ppi_file),
                             os.path.join(data_dir, alias_file),
                             universe_genes=od.gene_names,
                             confidence_threshold=config.string_threshold)

    records, histories, confusions = [], [], []
    for f in folds:
        tag = f"r{f.repeat}f{f.fold}"
        print(f"\n=== fold {tag} ===")
        prep = MultiOmicsTransformer(
            n_genes=od.n_genes, gene_names=od.gene_names, top_k=config.top_k,
            curated_genes=curated, active_modalities=("rna", "cnv", "methy"))
        prep.fit(od.X[f.train_idx], od.y[f.train_idx])
        selected = prep.get_selected_gene_names()
        n_sel = len(selected)
        Xtr = prep.transform(od.X[f.train_idx]); ytr = od.y[f.train_idx]
        Xte = prep.transform(od.X[f.test_idx]); yte = od.y[f.test_idx]

        # per-fold graph -> PE + SPD computed ONCE (fold-constants)
        A = torch.tensor(cache.induced_adjacency(selected), dtype=torch.float32).to(device)
        assert A.shape == (n_sel, n_sel), f"graph {A.shape} != N={n_sel}"
        graph_pe = GraphPositionalEncoding(pe_dim=config.pe_dim, method=config.pe_method)(A)
        spd = compute_shortest_path_matrix(A, max_dist=config.max_dist).to(device)

        inner_tr, inner_val = train_test_split(
            np.arange(len(ytr)), test_size=config.inner_val_frac,
            stratify=ytr, random_state=config.seed)
        cw = torch.tensor(sqrt_dampened_weights(ytr[inner_tr], num_classes), dtype=torch.float32)
        criterion = MultiClassFocalLoss(alpha=cw, gamma=config.focal_gamma, reduction="mean")

        rtr, ctr, mtr = split_modalities(Xtr[inner_tr], n_sel)
        rva, cva, mva = split_modalities(Xtr[inner_val], n_sel)
        rte, cte, mte = split_modalities(Xte, n_sel)
        tr_loader = DataLoader(DS(rtr, ctr, mtr, ytr[inner_tr]),
                               batch_size=config.batch_size, shuffle=True, drop_last=True)
        va_loader = DataLoader(DS(rva, cva, mva, ytr[inner_val]),
                               batch_size=config.batch_size, shuffle=False)
        te_loader = DataLoader(DS(rte, cte, mte, yte),
                               batch_size=config.batch_size, shuffle=False)

        model = MultiOmicsGraphClassifier(
            num_classes=num_classes, d=config.d, pe_dim=config.pe_dim,
            mini_heads=config.mini_heads, global_heads=config.global_heads,
            global_layers=config.global_layers, dropout=config.dropout,
            rna_dropout_prob=config.rna_dropout, meth_dropout_prob=config.meth_dropout,
            cnv_dropout_prob=config.cnv_dropout, max_dist=config.max_dist,
            attention_mode=config.attention_mode)
        trainer = FoldTrainer(model, tr_loader, va_loader, device, spd, graph_pe,
                              criterion, config.lr, config.min_lr, config.weight_decay)
        trainer.fit_early(config.max_epochs, config.patience, config.log_every)

        proba, ytrue = trainer.predict_proba(te_loader)
        pred = proba.argmax(1)
        m = compute_fold_metrics(ytrue, pred, proba, labels_idx, class_names)
        m.update({"model": "MOGFormer_BASELINE", "repeat": f.repeat, "fold": f.fold,
                  "n_selected": n_sel, "best_inner_f1": trainer.best_val_f1})
        records.append(m)
        histories.append({k: list(v) for k, v in trainer.history.items()})
        confusions.append(fold_confusion(ytrue, pred, labels_idx))
        print(f"  {tag}: macro-F1={m['macro_f1']:.4f} | "
              f"inner-best={trainer.best_val_f1:.4f}@e{trainer.best_epoch} | "
              f"best-losses train={trainer.best_train_loss:.4f} "
              f"val={trainer.best_val_loss:.4f} | "
              f"stopped at epoch {trainer.stopped_epoch}/{config.max_epochs}")

    _write_baseline_artifacts(config, records, class_names, results_dir, fig_dir,
                              histories, confusions)
    return records


def _write_baseline_artifacts(config, records, class_names, results_dir, fig_dir,
                              histories=None, confusions=None):

    wide = pd.DataFrame(records)
    wide.to_csv(os.path.join(results_dir, "metrics_per_fold.csv"), index=False)
    agg = aggregate(wide["macro_f1"].to_numpy())
    with open(os.path.join(results_dir, "config.yaml"), "w") as fh:
        yaml.safe_dump({"config": _config_to_dict(config),
                        "baseline_macro_f1": {k: float(v) if v == v else None
                                              for k, v in agg.items()}}, fh, sort_keys=False)
    with open(os.path.join(results_dir, "metrics_summary.md"), "w") as fh:
        fh.write("# MOGFormer BASELINE (STEP 0)\n\n")
        fh.write(f"- macro-F1: **{agg['mean']:.4f}** "
                 f"(95% CI [{agg['ci95_lo']:.4f}, {agg['ci95_hi']:.4f}], "
                 f"NB CI [{agg['nb_ci95_lo']:.4f}, {agg['nb_ci95_hi']:.4f}], n={agg['n']})\n")
        fh.write("- per-class F1 (mean):\n")
        for c in class_names:
            if f"f1__{c}" in wide:
                fh.write(f"  - {c}: {wide[f'f1__{c}'].mean():.4f}\n")
    plot_c00(agg["mean"], (agg["ci95_lo"], agg["ci95_hi"]), fig_dir)
    fold_plots.save_monitoring_figures(wide, histories or [], confusions or [],
                                       class_names, fig_dir)
    print(f"\nBASELINE macro-F1 = {agg['mean']:.4f} "
          f"[{agg['ci95_lo']:.4f}, {agg['ci95_hi']:.4f}]  (target 0.852, ref 0.738)")


def plot_c00(mean, ci, fig_dir, xgb=0.852, pam50=0.738):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.bar(["MOGFormer\nBASELINE"], [mean],
           yerr=[[mean - ci[0]], [ci[1] - mean]], capsize=6, color="#4C72B0")
    ax.axhline(xgb, ls="--", color="#C44E52", label=f"XGBoost {xgb:.3f}")
    ax.axhline(pam50, ls=":", color="#555555", label=f"PAM50-centroid {pam50:.3f}")
    ax.text(0, mean + 0.01, f"{mean:.3f}", ha="center", fontweight="bold")
    ax.set_ylim(0, 1); ax.set_ylabel("macro-F1")
    ax.set_title("C00 - MOGFormer baseline vs references"); ax.legend(loc="lower right")
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(fig_dir, f"C00_baseline_vs_references.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
