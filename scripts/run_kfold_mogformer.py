"""
scripts/run_kfold_mogformer.py — STEP 0 entry point (run from project root):
    python -m scripts.run_kfold_mogformer

No injection: the runner imports the model + graph + loss directly from src/.
For a single-fold dev run (replaces train.py): set max_folds=1.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.runner import MOGFormerConfig, run_mogformer_cv

DATA_DIR = "data/raw"               # data_*.csv + the two STRING files
FOLDS    = "data/raw/folds.json"     # the EXACT folds.json from Phase 0 (pairing!)
CURATED  = "data/raw/CGenes.txt"
RESULTS  = "results/"

if __name__ == "__main__":
    cfg = MOGFormerConfig(pe_method = "rwpe",              # "rwpe" | "laplacian"
                      max_epochs = 300, 
                      patience = 100,
                      max_folds = None,                 # 1 -> single fold | None -> 5 folds 
                      n_repeats_used = 1,               # 5 -> full 25 estimates | 1 -> single repeat (5 folds
                      gene_id_embedding = True,
                      pretrained_gene_emb = "data/raw/FROGS-ARCHS4/FROGS-ARCHS4", # "path" -> pre-trained emb | "" -> learnt
                      pretrained_emb_adapter_rank = 64,
                      attention_bias_mode = "inside",   # "inside" -> softmax(QK+B)V | "dual" -> softmax(QK)V+softmax(B)V
                      numerical_tokenizer = "mlp",
                      attention_lambda_gate = False,      # True -> A = softmax(QK + λB)V | False -> λ=1
                      unimodal_dropout_fill= "mask_token" # "zero"-> 0 fill dropout | "mask_token" -> mask token fill 
                      )                                 # exclude_classes = ("BRCA_Normal",) to restrict 4-class
                            
    # quick dev run instead:  cfg = MOGFormerConfig(max_folds=1, max_epochs=3)
    run_mogformer_cv(
        config=cfg, 
        data_dir=DATA_DIR, 
        results_dir=RESULTS, 
        folds_path=FOLDS,
        curated_genes_file=CURATED,
    )
