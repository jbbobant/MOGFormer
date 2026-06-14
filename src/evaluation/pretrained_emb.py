import numpy as np
import pandas as pd
from typing import List, Optional


def load_archs4_embeddings(
    symbollist_path: str,
    emb_path: str,
    universe_genes: List[str],
    cache_path: Optional[str] = None,
) -> np.ndarray:
    """Load FROG-ARCHS4 embeddings aligned to the gene universe.

    Args:
        symbollist_path : FROG-ARCHS4_symbollist.txt (one HGNC symbol per line,
                          same row order as emb.csv)
        emb_path        : FROG-ARCHS4_emb.csv (rows = genes, cols = dims,
                          no header, no index)
        universe_genes  : ordered HGNC symbols from load_omics() / od.gene_names
        cache_path      : optional path to save/load the aligned (n_universe, d_e)
                          .npy matrix — skips re-reading CSV on subsequent runs

    Returns:
        float32 ndarray of shape (n_universe, d_e).
        Genes absent from ARCHS4 are zero-filled and flagged.
    """
    n = len(universe_genes)

    # --- 0. Cache hit ---
    if cache_path:
        try:
            mat = np.load(cache_path)
            if mat.shape[0] == n:
                print(f"[archs4] loaded from cache {cache_path}: {mat.shape}")
                return mat.astype(np.float32)
        except Exception:
            pass

    # --- 1. Load symbol list ---
    with open(symbollist_path) as f:
        symbols = [ln.strip() for ln in f
                   if ln.strip() and not ln.startswith("#")]
    print(f"[archs4] {len(symbols)} symbols in symbollist")

    # --- 2. Load embedding matrix ---
    try:
        emb_df = pd.read_csv(emb_path, header=None, index_col=None)
    except Exception:
        emb_df = pd.read_csv(emb_path, header=None, index_col=None, sep="\t")
    emb_array = emb_df.to_numpy(dtype=np.float32)
    assert len(emb_array) == len(symbols), (
        f"symbollist has {len(symbols)} entries but emb has {len(emb_array)} rows")
    d_e = emb_array.shape[1]
    print(f"[archs4] embedding matrix: {emb_array.shape}")

    # --- 3. Build symbol -> row index (first occurrence wins) ---
    symbol_to_row = {}
    for row_idx, sym in enumerate(symbols):
        if sym not in symbol_to_row:
            symbol_to_row[sym] = row_idx

    # --- 4. Align to universe ---
    universe_set = set(universe_genes)
    in_archs4 = sum(1 for s in universe_genes if s in symbol_to_row)
    print(f"[archs4] universe overlap: {in_archs4}/{n} genes found in ARCHS4 "
          f"({n - in_archs4} will be zero-filled)")

    emb_matrix = np.zeros((n, d_e), dtype=np.float32)
    for col_idx, sym in enumerate(universe_genes):
        if sym in symbol_to_row:
            emb_matrix[col_idx] = emb_array[symbol_to_row[sym]]

    # --- 5. L2-normalise non-zero rows ---
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    emb_matrix = emb_matrix / norms

    # --- 6. Cache ---
    if cache_path:
        np.save(cache_path, emb_matrix)
        print(f"[archs4] saved aligned matrix to {cache_path}")

    return emb_matrix