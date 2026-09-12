#!/usr/bin/env python
# ============================================================================
# Phase 4 / STEP 0 — CollecTRI GRN: download, preprocess, coverage diagnostics
# ----------------------------------------------------------------------------
# NO TRAINING. This script:
#   (1) downloads CollecTRI (directed, signed TF->target) via decoupleR/OmniPath
#   (2) preprocesses to clean signed edges (drop sign==0, drop sign-ambiguous)
#   (3) saves a deterministic offline file (read by training runs later)
#   (4) computes coverage at TWO denominators required by spec section 2c:
#         - FULL universe  = genes common to all 3 omics  (od.gene_names)
#         - CURATED set     = CGenes.txt filtered to universe
#
# Run this ONCE where OmniPath has network access (local box, or a Kaggle
# session with internet ON for this single run). Training reads the saved file.
#
# All gene references are HGNC SYMBOLS (matches STRING induction + G_i embedding).
# ============================================================================

import os
import sys
import json
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# CONFIG — point these at YOUR files. Defaults mirror your Kaggle paths.
# ----------------------------------------------------------------------------
# The full universe = genes shared across RNA/CNV/methy. The cheapest robust
# way to reproduce it WITHOUT re-running your loader is to point at the three
# omics matrices and intersect their indices exactly as align_omics does.
# If you already have the aligned gene list saved, set UNIVERSE_TXT instead.

RNA_CSV     = "data/raw/data_rna_seq_v2_rsem.csv"
CNV_CSV     = "data/raw/data_cnv.csv"
METHY_CSV   = "data/raw/data_methylation_M.csv"

CURATED_TXT = "data/raw/CGenes.txt"

OUT_DIR     = "data/processed/grn_step0_out"
os.makedirs(OUT_DIR, exist_ok=True)

# saved artifacts
GRN_PARQUET = os.path.join(OUT_DIR, "collectri_signed_edges.parquet")  # the offline GRN training reads
GRN_TSV     = os.path.join(OUT_DIR, "collectri_signed_edges.tsv")      # human-readable mirror
COVERAGE_JSON = os.path.join(OUT_DIR, "G0_grn_coverage.json")


# ----------------------------------------------------------------------------
# 1. DOWNLOAD CollecTRI
# ----------------------------------------------------------------------------
def download_collectri() -> pd.DataFrame:
    """
    Returns a DataFrame with columns: source (TF), target, weight (+1/-1), n_sources.
    Tries the modern decoupler API first, then legacy, then a raw OmniPath pull.
    All symbol-based, human (organism=9606).
    """
    # --- attempt A: modern decoupler (>=1.6) op.collectri --------------------
    try:
        import decoupler as dc
        try:
            net = dc.op.collectri(organism="human")          # newest API
        except AttributeError:
            net = dc.get_collectri(organism="human", split_complexes=False)  # legacy API
        net = net.rename(columns={"source": "source", "target": "target", "weight": "weight"})
        print(f"[download] decoupler CollecTRI: {len(net)} raw edges")
        # n_sources may live in a 'PMID' / 'resources' col depending on version; best-effort
        if "PMID" in net.columns:
            net["n_sources"] = net["PMID"].fillna("").map(lambda s: len({p for p in str(s).split(";") if p}))
        else:
            net["n_sources"] = np.nan
        return net[["source", "target", "weight", "n_sources"]].copy()
    except Exception as e:
        print(f"[download] decoupler path failed ({e}); falling back to omnipath raw pull")

    # --- attempt B: raw omnipath ---------------------------------------------
    try:
        import omnipath as op
        net = op.interactions.CollecTRI.get(genesymbols=True, organism="human")
        # omnipath CollecTRI columns: source_genesymbol, target_genesymbol,
        # is_stimulation, is_inhibition, ...
        net = net.rename(columns={
            "source_genesymbol": "source",
            "target_genesymbol": "target",
        })
        sign = net["is_stimulation"].astype(int) - net["is_inhibition"].astype(int)
        net["weight"] = sign
        if "n_references" in net.columns:
            net["n_sources"] = net["n_references"]
        elif "references" in net.columns:
            net["n_sources"] = net["references"].fillna("").map(
                lambda s: len({p for p in str(s).split(";") if p}))
        else:
            net["n_sources"] = np.nan
        print(f"[download] omnipath CollecTRI: {len(net)} raw edges")
        return net[["source", "target", "weight", "n_sources"]].copy()
    except Exception as e:
        print(f"[download] omnipath path also failed ({e})")
        raise SystemExit(
            "Could not download CollecTRI. Install one of:\n"
            "  pip install decoupler\n"
            "  pip install omnipath\n"
            "and run this script with internet access enabled.")


# ----------------------------------------------------------------------------
# 2. PREPROCESS -> clean signed edges
#    - keep only cleanly signed: weight in {+1,-1}; drop weight==0 / NaN
#    - drop SIGN-AMBIGUOUS pairs: a (TF,target) appearing with BOTH +1 and -1
#    - drop self-loops
#    - dedup
# ----------------------------------------------------------------------------
def preprocess(net: pd.DataFrame) -> pd.DataFrame:
    n0 = len(net)
    net = net.dropna(subset=["source", "target", "weight"]).copy()
    net["source"] = net["source"].astype(str)
    net["target"] = net["target"].astype(str)

    # binarize sign; keep only clean +1/-1
    net["sign"] = np.sign(net["weight"]).astype(int)
    net = net[net["sign"].isin([-1, 1])]
    n_signed = len(net)

    # drop self-loops
    net = net[net["source"] != net["target"]]

    # collapse exact duplicate (source,target,sign)
    net = net.drop_duplicates(subset=["source", "target", "sign"])

    # detect & drop sign-ambiguous (same TF->target with both +1 and -1)
    pair_sign_counts = net.groupby(["source", "target"])["sign"].nunique()
    ambiguous = set(pair_sign_counts[pair_sign_counts > 1].index)
    if ambiguous:
        before = len(net)
        net = net[~net.set_index(["source", "target"]).index.isin(ambiguous)].copy()
        print(f"[preprocess] dropped {before - len(net)} edges from "
              f"{len(ambiguous)} sign-ambiguous TF->target pairs")

    net = net[["source", "target", "sign", "n_sources"]].reset_index(drop=True)
    print(f"[preprocess] {n0} raw -> {n_signed} signed -> {len(net)} clean directed signed edges")
    print(f"[preprocess] sign balance: "
          f"{(net['sign'] == 1).sum()} activation / {(net['sign'] == -1).sum()} repression")
    return net


def save_grn(net: pd.DataFrame):
    net.to_parquet(GRN_PARQUET, index=False)
    net.to_csv(GRN_TSV, sep="\t", index=False)
    print(f"[save] {GRN_PARQUET}")
    print(f"[save] {GRN_TSV}")


# ----------------------------------------------------------------------------
# 3. UNIVERSE + CURATED loading (mirror your codebase exactly)
# ----------------------------------------------------------------------------
def load_universe() -> list:
    
    def _index(path):
        df = pd.read_csv(path, index_col=0, nrows=0)  # header only? no -- need row index
        # we only need the row index (genes); read just the first column quickly
        idx = pd.read_csv(path, index_col=0, usecols=[0]).index
        return set(map(str, idx))
    for p in (RNA_CSV, CNV_CSV, METHY_CSV):
        if not os.path.exists(p):
            raise SystemExit(f"[universe] missing omics file {p}; set UNIVERSE_TXT instead.")
    rna_i, cnv_i, methy_i = _index(RNA_CSV), _index(CNV_CSV), _index(METHY_CSV)
    uni = sorted(rna_i & cnv_i & methy_i)
    print(f"[universe] {len(uni)} genes common to all three omics "
          f"(RNA {len(rna_i)}, CNV {len(cnv_i)}, methy {len(methy_i)})")
    return uni


def load_curated(universe: set) -> list:
    if not os.path.exists(CURATED_TXT):
        print(f"[curated] file not found at {CURATED_TXT}; curated coverage skipped")
        return []
    with open(CURATED_TXT) as f:
        raw = [ln.strip() for ln in f if ln.strip()]
    present = [g for g in raw if g in universe]
    print(f"[curated] {len(present)}/{len(raw)} curated genes present in universe")
    return present


# ----------------------------------------------------------------------------
# 4. COVERAGE DIAGNOSTICS (spec 2c) at a given denominator (gene set)
# ----------------------------------------------------------------------------
def directed_path_coverage_4hop(genes: list, edges_df: pd.DataFrame, max_hops: int = 4) -> float:
    """Fraction of ORDERED pairs (i,j), i!=j, with a directed path TF-style
    within max_hops, on the induced directed graph. Expected MUCH lower than
    STRING (confirms adjacency, not directed-SPD). Uses BFS over a sparse adj."""
    gset = set(genes)
    sub = edges_df[edges_df["source"].isin(gset) & edges_df["target"].isin(gset)]
    if len(sub) == 0 or len(genes) < 2:
        return 0.0
    # adjacency list
    adj = {}
    for s, t in zip(sub["source"].to_numpy(), sub["target"].to_numpy()):
        adj.setdefault(s, []).append(t)
    n = len(genes)
    total_ordered = n * (n - 1)
    reachable = 0
    from collections import deque
    for src in genes:
        if src not in adj:
            continue
        seen = {src: 0}
        dq = deque([src])
        while dq:
            u = dq.popleft()
            if seen[u] >= max_hops:
                continue
            for v in adj.get(u, []):
                if v not in seen:
                    seen[v] = seen[u] + 1
                    dq.append(v)
        reachable += sum(1 for v in seen if v != src)
    return reachable / total_ordered if total_ordered else 0.0


def coverage_report(name: str, genes: list, edges_df: pd.DataFrame) -> dict:
    gset = set(genes)
    n = len(genes)

    sub = edges_df[edges_df["source"].isin(gset) & edges_df["target"].isin(gset)].copy()
    n_edges = len(sub)

    tfs_out      = set(sub["source"].unique())          # have outgoing edges (regulators)
    targets_in   = set(sub["target"].unique())          # have incoming edges
    annotated    = tfs_out | targets_in                 # any GRN annotation within the set
    target_only  = targets_in - tfs_out

    n_act  = int((sub["sign"] == 1).sum())
    n_repr = int((sub["sign"] == -1).sum())

    # edge density relative to a directed complete graph
    density = n_edges / (n * (n - 1)) if n > 1 else 0.0

    path_cov = directed_path_coverage_4hop(genes, edges_df, max_hops=4)

    rep = {
        "denominator": name,
        "n_genes": n,
        "n_genes_with_any_grn_annotation": len(annotated),
        "frac_genes_annotated": round(len(annotated) / n, 4) if n else 0.0,
        "n_TFs_with_outgoing": len(tfs_out),
        "frac_TFs": round(len(tfs_out) / n, 4) if n else 0.0,
        "n_target_only": len(target_only),
        "frac_target_only": round(len(target_only) / n, 4) if n else 0.0,
        "n_directed_edges_within_set": n_edges,
        "edge_density_directed": round(density, 6),
        "n_activation": n_act,
        "n_repression": n_repr,
        "frac_activation": round(n_act / n_edges, 4) if n_edges else 0.0,
        "directed_path_coverage_4hop": round(path_cov, 4),
    }

    # SPEC 2c DECISION GATE
    rep["decision_gate_30pct"] = (
        "OK (>=30% annotated -> adjacency_signed is comfortably supported)"
        if rep["frac_genes_annotated"] >= 0.30 else
        "SPARSE (<30% annotated -> adjacency_signed ONLY; defer L5/L6; expect weak/null signal)"
    )
    return rep


def print_report(rep: dict):
    print(f"\n{'='*64}\nCOVERAGE — {rep['denominator']}  (N={rep['n_genes']})\n{'='*64}")
    print(f"  genes with ANY GRN edge   : {rep['n_genes_with_any_grn_annotation']:5d}  "
          f"({rep['frac_genes_annotated']*100:.1f}%)")
    print(f"  TFs (outgoing edges)      : {rep['n_TFs_with_outgoing']:5d}  "
          f"({rep['frac_TFs']*100:.1f}%)")
    print(f"  target-only (incoming)    : {rep['n_target_only']:5d}  "
          f"({rep['frac_target_only']*100:.1f}%)")
    print(f"  directed edges within set : {rep['n_directed_edges_within_set']:5d}  "
          f"(density {rep['edge_density_directed']:.2e})")
    print(f"  sign balance              : {rep['n_activation']} act / "
          f"{rep['n_repression']} repr  ({rep['frac_activation']*100:.1f}% act)")
    print(f"  directed 4-hop path cover : {rep['directed_path_coverage_4hop']*100:.2f}%  "
          f"(cf. STRING ~near-complete -> confirms adjacency, not directed-SPD)")
    print(f"  >>> DECISION: {rep['decision_gate_30pct']}")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    print("STEP 0 — CollecTRI GRN download / preprocess / coverage (NO TRAINING)\n")

    # 1-2. download + preprocess (skip re-download if file already exists)
    if os.path.exists(GRN_PARQUET):
        print(f"[cache] reusing existing {GRN_PARQUET} (delete to force re-download)")
        net = pd.read_parquet(GRN_PARQUET)
    else:
        raw = download_collectri()
        net = preprocess(raw)
        save_grn(net)

    # 3. universe + curated
    universe = load_universe()
    uni_set = set(universe)
    curated = load_curated(uni_set)

    # ID-ALIGNMENT SANITY (spec 8): are GRN symbols even in the same namespace?
    grn_nodes = set(net["source"]) | set(net["target"])
    overlap_uni = len(grn_nodes & uni_set)
    print(f"\n[id-check] GRN nodes overlapping the universe: {overlap_uni} "
          f"(GRN has {len(grn_nodes)} distinct symbols). "
          f"{'OK' if overlap_uni > 0 else 'NAMESPACE MISMATCH -- investigate symbol mapping!'}")

    # 4. coverage at both denominators
    reports = {}
    rep_full = coverage_report("FULL_universe", universe, net)
    print_report(rep_full); reports["full"] = rep_full

    if curated:
        rep_cur = coverage_report("CURATED_set", curated, net)
        print_report(rep_cur); reports["curated"] = rep_cur

    with open(COVERAGE_JSON, "w") as fh:
        json.dump(reports, fh, indent=2)
    print(f"\n[save] {COVERAGE_JSON}")
    print("\nSTEP 0 complete. Inspect coverage above BEFORE deciding to build the L2 attention term.")


if __name__ == "__main__":
    main()