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
    cfg = MOGFormerConfig(max_folds=1, max_epochs=3)         # full 25-estimate BASELINE
    # quick dev run instead:  cfg = MOGFormerConfig(max_folds=1, max_epochs=3)
    run_mogformer_cv(
        config=cfg, data_dir=DATA_DIR, results_dir=RESULTS, folds_path=FOLDS,
        curated_genes_file=CURATED,
    )
