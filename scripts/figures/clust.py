import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
 
# ----------------------------------------------------------------------------- CONFIG
EMB_PATH   = r"results/SSL/SURV_CLEAN/E600_p60_gated_meta_rna_meth/C_LumA_MOFA.parquet"        # <-- edit
CLIN_PATH  = r"data/clean/survival_features_common.csv"     # <-- edit
OUT_DIR    = r"results/SSL/SURV_CLEAN/E600_p60_gated_meta_rna_meth/MOFA"
ID_COL     = "patient_id"               # clinical id column (matches nuisance_check)
EMB_ID_COL = "patient_id"            # id column inside the parquet (matches extract code)
RECORDED_PAC_K2 = 0.072              # the value you reported for U2 K=2 (checksum target)
RECORDED_SIZES  = (141, 339)         # expected K=2 sizes (panel B); order-insensitive
RUN_CHECKSUM = True                  # set False to skip the 200-resample PAC reproduction
 
AGE_COL = "clinical__demographic.age_at_index"
TECH_COLS = [
    "clinical__demographic.race",
    "clinical__demographic.ethnicity",
    "clinical__diagnoses.ajcc_staging_system_edition",
    "clinical__diagnoses.year_of_diagnosis",
    "clinical__diagnoses.classification_of_tumor",
    "clinical__demographic.age_at_index",
]
BIO_COLS = [
    "clinical__diagnoses.ajcc_pathologic_stage",
    "clinical__diagnoses.ajcc_pathologic_n",
    "clinical__diagnoses.ajcc_pathologic_m",
]
ALL_COLS = TECH_COLS + BIO_COLS
 
MISSING_TOKENS = {"", "na", "nan", "none", "not reported", "unknown",
                  "[not available]", "[unknown]", "[not applicable]",
                  "not available", "--", "'--", "[discrepancy]"}
 
# ----------------------------------------------------------------------------- helpers
def _clean_cat(s):
    """Map GDC missing tokens to NaN, keep the rest as strings."""
    out = s.astype(str).str.strip()
    return out.where(~out.str.lower().isin(MISSING_TOKENS), other=np.nan)
 
def _collapse_stage(s):
    v = s.astype(str).str.upper().str.replace(" ", "", regex=False)
    out = pd.Series(np.nan, index=s.index, dtype=object)
    for tag in ["IV", "III", "II", "I"]:                 # longest-first
        hit = v.str.match("STAGE" + tag + r"($|[ABC])")  # allow A/B/C suffix, no \b
        out[hit & out.isna()] = "Stage " + tag
    return out
 
def _collapse_TN(s, letter):
    v = s.astype(str).str.upper()
    out = pd.Series(np.nan, index=s.index, dtype=object)
    for d in ["3", "2", "1", "0"]:
        hit = v.str.match(letter + d)
        out[hit & out.isna()] = letter + d
    return out
 
def cramers_v(ct):
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    k = min(ct.shape) - 1
    return np.sqrt(chi2 / (n * k)) if (n > 0 and k > 0) else np.nan
 
def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order-preserving."""
    items = [(k, v) for k, v in pvals.items() if v is not None and np.isfinite(v)]
    m = len(items)
    order = sorted(items, key=lambda kv: kv[1])
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(order):
        running = max(running, (m - i) * p)
        adj[k] = min(running, 1.0)
    for k, v in pvals.items():
        adj.setdefault(k, np.nan)
    return adj
 
def savefig(fig, name):
    fig.savefig(os.path.join(OUT_DIR, f"{name}.{"png"}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
 
C_SMALL, C_LARGE = "#0072B2", "#E69F00"   # Okabe-Ito, colorblind-safe
 
# ----------------------------------------------------------------------------- load
def load_embedding(path):
    df = pd.read_parquet(path)
    if EMB_ID_COL not in df.columns:
        raise ValueError(f"'{EMB_ID_COL}' not in parquet columns: {list(df.columns)[:6]}...")
    emb_cols = sorted([c for c in df.columns if c.startswith("c_")])
    if not emb_cols:
        emb_cols = sorted([c for c in df.columns if c != EMB_ID_COL and
                           pd.api.types.is_numeric_dtype(df[c])])
    X = df[emb_cols].to_numpy(dtype=np.float64)
    pid = df[EMB_ID_COL].astype(str).tolist()
    print(f"[load] embedding {X.shape} from {len(emb_cols)} dims; {len(pid)} patients")
    return X, pid
 
# ----------------------------------------------------------------------------- (2) checksum
def reproduce_pac_k2(X, n_resample=200, frac=0.8, seed=42, pac_bounds=(0.1, 0.9)):
    """EXACT mirror of consensus_pac for K=2. In the original, a single
    rng=default_rng(42) is consumed across K_list=(2,3,4,5,6) with K=2 FIRST, so a
    fresh rng(42) run of K=2 alone reproduces the same RNG stream => same PAC."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    lo, hi = pac_bounds
    K = 2
    co = np.zeros((n, n)); cnt = np.zeros((n, n))
    for _ in range(n_resample):
        idx = rng.choice(n, int(frac * n), replace=False)             # rng.choice FIRST
        lbl = KMeans(n_clusters=K, n_init=5,
                     random_state=int(rng.integers(1e6))).fit_predict(X[idx])  # then integers
        oh = np.zeros((len(idx), K)); oh[np.arange(len(idx)), lbl] = 1
        block = oh @ oh.T
        co[np.ix_(idx, idx)] += block
        cnt[np.ix_(idx, idx)] += 1
    M = np.divide(co, cnt, out=np.zeros_like(co), where=cnt > 0)
    offdiag = M[np.triu_indices(n, k=1)]
    pac = float(np.mean((offdiag > lo) & (offdiag < hi)))
    return pac, M
 
# ----------------------------------------------------------------------------- (3) partition
def reproduce_labels_k2(X):
    """EXACT mirror of phase2_acceptance_diagnostics: KMeans on RAW X."""
    km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X)
    lab = km.labels_
    sizes = np.bincount(lab)
    # canonicalize by size: A = smaller cluster, B = larger (robust to 0/1 label swap)
    small = int(np.argmin(sizes))
    canon = np.where(lab == small, 0, 1)   # 0 = small cluster (A), 1 = large (B)
    return canon, tuple(np.bincount(canon).tolist())
 
# ----------------------------------------------------------------------------- (4) tests
def is_continuous(s):
    return pd.api.types.is_numeric_dtype(s) and s.nunique() > 8   # same rule as _assoc_pc_var
 
def test_covariate(cluster, series):
    """Return dict with kind, effect size, p, and a display table."""
    s = series.copy()
    mask = s.notna().values
    c = cluster[mask]; s = s[mask]
    if len(np.unique(c)) < 2 or s.nunique() < 2 or len(c) < 5:
        return dict(kind="skip", eff=np.nan, p=np.nan, n=int(mask.sum()), note="")
    if is_continuous(s):
        x = c.astype(float); y = s.astype(float).values
        g0, g1 = y[x == 0], y[x == 1]
        U, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
        rb = 1.0 - 2.0 * U / (len(g0) * len(g1))                    # rank-biserial
        pb, _ = stats.pointbiserialr(x, y)                          # signed, panel-comparable
        note = f"median A={np.median(g0):.1f} vs B={np.median(g1):.1f}"
        return dict(kind="continuous", eff=abs(pb), signed=pb, rb=rb, p=float(p),
                    n=int(mask.sum()), note=note)
    ct = pd.crosstab(np.asarray(c), _clean_cat(s).values)   # positional, avoid index-align
    if ct.shape[1] < 2 or ct.values.sum() < 5:
        return dict(kind="skip", eff=np.nan, p=np.nan, n=int(mask.sum()), note="")
    if ct.shape == (2, 2):
        _, p = stats.fisher_exact(ct.values)
    else:
        p = stats.chi2_contingency(ct.values, correction=False)[1]
    return dict(kind="categorical", eff=cramers_v(ct), p=float(p),
                n=int(mask.sum()), note=f"{ct.shape[1]} levels", ct=ct)
 
# ----------------------------------------------------------------------------- (5) axis
def pc_axis_analysis(X, cluster, age, n_pcs=10):
    Xc = X - X.mean(0, keepdims=True)                               # mirror _top_pcs
    p = PCA(n_components=min(n_pcs, X.shape[1], X.shape[0] - 1)).fit(Xc)
    pcs = p.transform(Xc); evr = p.explained_variance_ratio_
    rows = []
    for i in range(pcs.shape[1]):
        pb, _ = stats.pointbiserialr(cluster.astype(float), pcs[:, i])
        am = age.notna().values
        if am.sum() > 5:
            rho, _ = stats.spearmanr(pcs[am, i], age[am].astype(float))
        else:
            rho = np.nan
        rows.append((i + 1, evr[i], abs(pb), abs(rho)))
    axis = pd.DataFrame(rows, columns=["PC", "evr", "abs_pb_cluster", "abs_rho_age"])
    return pcs, evr, axis
 
def age_surrogate_auc(cluster, age):
    am = age.notna().values
    if am.sum() < 20:
        return np.nan
    Xa = age[am].astype(float).values.reshape(-1, 1)
    ya = cluster[am]
    try:
        auc = cross_val_score(LogisticRegression(max_iter=1000), Xa, ya,
                              cv=5, scoring="roc_auc").mean()
    except Exception:
        auc = np.nan
    return float(auc)
 
# ----------------------------------------------------------------------------- figures
def fig_age(cluster, age):
    am = age.notna().values
    g0 = age[am & (cluster == 0)].astype(float); g1 = age[am & (cluster == 1)].astype(float)
    U, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
    rb = 1 - 2 * U / (len(g0) * len(g1))
    fig, ax = plt.subplots(figsize=(6, 5))
    parts = ax.violinplot([g0.values, g1.values], showextrema=False)
    for b, col in zip(parts["bodies"], [C_SMALL, C_LARGE]):
        b.set_facecolor(col); b.set_alpha(0.35)
    ax.boxplot([g0.values, g1.values], widths=0.15, showfliers=False,
               medianprops=dict(color="k"))
    for i, (g, col) in enumerate(zip([g0, g1], [C_SMALL, C_LARGE]), start=1):
        jit = np.random.default_rng(0).normal(i, 0.04, len(g))
        ax.scatter(jit, g.values, s=10, color=col, alpha=0.5, edgecolor="none")
    ax.set_xticks([1, 2]); ax.set_xticklabels([f"A (n={len(g0)})", f"B (n={len(g1)})"])
    ax.set_ylabel("age at index (years)")
    ax.set_title(f"Age by U2 K=2 cluster — MWU p={p:.2e}, rank-biserial={rb:+.2f}")
    savefig(fig, "gate_age_by_cluster")
 
def fig_effects(results):
    labels, effs, cats, stars = [], [], [], []
    for col in ALL_COLS:
        r = results.get(col)
        if r is None or r["kind"] == "skip":
            continue
        labels.append(col.split(".")[-1]); effs.append(r["eff"])
        cats.append("tech" if col in TECH_COLS else "bio")
        stars.append("*" if (r.get("holm", 1) < 0.05) else "")
    y = np.arange(len(labels))
    cols = ["#c0392b" if c == "tech" else "#1f9d55" for c in cats]
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(labels) + 1.5))
    ax.barh(y, effs, color=cols, alpha=0.85)
    for yi, e, st in zip(y, effs, stars):
        ax.text(e + 0.005, yi, st, va="center", fontsize=13)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.invert_yaxis(); ax.set_xlabel("effect size (|point-biserial| or Cramér's V)")
    ax.set_title("What the U2 K=2 split tracks\n(red=technical/demographic, green=biological; * Holm p<0.05)")
    savefig(fig, "gate_effect_sizes")
 
def fig_composition(cluster, clin):
    specs = [("clinical__diagnoses.ajcc_pathologic_stage", _collapse_stage, "stage"),
             ("clinical__diagnoses.ajcc_pathologic_n", lambda s: _collapse_TN(s, "N"), "N"),
             ("clinical__diagnoses.ajcc_pathologic_m", lambda s: _collapse_TN(s, "M"), "M")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (col, fn, title) in zip(axes, specs):
        if col not in clin.columns:
            ax.set_visible(False); continue
        cat = fn(clin[col])
        ct = pd.crosstab(np.asarray(cluster), cat.values, normalize="index")
        if ct.shape[1] == 0:
            ax.set_visible(False); continue
        bottom = np.zeros(len(ct))
        for lvl in ct.columns:
            ax.bar(["A", "B"], ct[lvl].values, bottom=bottom, label=str(lvl))
            bottom += ct[lvl].values
        ax.set_title(f"{title} composition by cluster"); ax.set_ylabel("proportion")
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); savefig(fig, "gate_stage_composition")
 
def fig_pc_axis(pcs, evr, axis, cluster, age):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    a = axes[0]
    comps = np.arange(1, len(evr) + 1)
    a.bar(comps, evr, color="#4c72b0", alpha=0.8)
    a.plot(comps, np.cumsum(evr), color="#C44E52", marker="o", lw=2)
    a.set_title(f"scree — PC1={evr[0]*100:.1f}%  (checksum ~13.7% for U2)")
    a.set_xlabel("PC"); a.set_ylabel("explained var ratio")
    b = axes[1]
    w = 0.4
    b.bar(axis["PC"] - w/2, axis["abs_pb_cluster"], w, label="|cluster ~ PC|", color="#333")
    b.bar(axis["PC"] + w/2, axis["abs_rho_age"], w, label="|age ~ PC|", color="#c0392b")
    b.set_xlabel("PC"); b.set_ylabel("|association|")
    b.set_title("does the split-axis coincide with the age-axis?"); b.legend(frameon=False)
    c = axes[2]
    am = age.notna().values
    sc = c.scatter(pcs[am, 0], pcs[am, 1], c=age[am].astype(float), cmap="viridis", s=14)
    edge = np.where(cluster[am] == 0, C_SMALL, C_LARGE)
    c.scatter(pcs[am, 0], pcs[am, 1], facecolors="none", edgecolors=edge, s=26, linewidths=0.6)
    fig.colorbar(sc, ax=c, label="age"); c.set_xlabel("PC1"); c.set_ylabel("PC2")
    c.set_title("PC1–PC2: fill=age, ring=cluster")
    fig.tight_layout(); savefig(fig, "gate_pc_axis")
 
# ----------------------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X, pid = load_embedding(EMB_PATH)
    n = X.shape[0]
 
    # (2) checksum
    if RUN_CHECKSUM:
        pac, _ = reproduce_pac_k2(X)
        dpac = abs(pac - RECORDED_PAC_K2)
        verdict = ("MATCH (same embedding)" if dpac < 0.03 else
                   "CLOSE (likely BLAS/sklearn numerics)" if dpac < 0.06 else
                   "MISMATCH -- different embedding or preprocessing; investigate")
        print(f"[checksum] reproduced K=2 PAC={pac:.3f}  vs recorded {RECORDED_PAC_K2:.3f}"
              f"  (|Δ|={dpac:.3f}) -> {verdict}")
    else:
        print("[checksum] skipped")
 
    # (3) partition
    cluster, sizes = reproduce_labels_k2(X)
    ok = (tuple(sorted(sizes)) == tuple(sorted(RECORDED_SIZES)))
    print(f"[partition] K=2 sizes A/B = {sizes}  (expected {RECORDED_SIZES}) -> "
          f"{'MATCH' if ok else 'DIFFERS -- check embedding/sklearn version'}")
 
    # (4) join clinical
    clin = pd.read_csv(CLIN_PATH)
    clin[ID_COL] = clin[ID_COL].astype(str)
    clin = clin.drop_duplicates(subset=ID_COL).set_index(ID_COL)
    aligned = clin.reindex([str(p) for p in pid])
    matched = int(aligned.notna().any(axis=1).sum())
    print(f"[join] {matched}/{n} patients matched on '{ID_COL}'"
          + ("" if matched == n else "  <-- MISMATCH: check barcode format"))
 
    age = aligned[AGE_COL] if AGE_COL in aligned.columns else pd.Series(np.nan, index=aligned.index)
    if age.notna().sum() and np.nanmedian(age.astype(float)) > 150:   # days -> years guard
        age = age.astype(float) / 365.25
        print("[age] detected days; converted to years")
 
    # covariate tests + Holm
    results, praw = {}, {}
    for col in ALL_COLS:
        if col not in aligned.columns:
            warnings.warn(f"[gate] column '{col}' missing; skipped."); continue
        r = test_covariate(cluster, aligned[col])
        results[col] = r
        if r["kind"] != "skip":
            praw[col] = r["p"]
    hp = holm(praw)
    for col, a in hp.items():
        results[col]["holm"] = a
 
    # (5) axis
    pcs, evr, axis = pc_axis_analysis(X, cluster, age)
    auc = age_surrogate_auc(cluster, age)
    split_pc = int(axis.loc[axis["abs_pb_cluster"].idxmax(), "PC"])
    age_pc = int(axis.loc[axis["abs_rho_age"].idxmax(), "PC"])
 
    # dumps
    rowout = []
    for col in ALL_COLS:
        r = results.get(col, {})
        if not r or r.get("kind") == "skip":
            continue
        rowout.append(dict(covariate=col, kind=r["kind"], effect=r["eff"],
                           p_raw=r["p"], p_holm=r.get("holm"), n=r["n"], note=r["note"]))
    pd.DataFrame(rowout).to_csv(os.path.join(OUT_DIR, "gate_association_table.csv"), index=False)
    axis.to_csv(os.path.join(OUT_DIR, "gate_pc_association.csv"), index=False)
    pd.DataFrame({"patient_id": pid, "cluster_AB": np.where(cluster == 0, "A", "B"),
                  "age": age.values}).to_csv(os.path.join(OUT_DIR, "gate_partition.csv"), index=False)
 
    # figures
    if age.notna().sum() > 5:
        fig_age(cluster, age)
        fig_pc_axis(pcs, evr, axis, cluster, age)
    fig_effects(results)
    fig_composition(cluster, aligned)
 
    # verdict
    age_r = results.get(AGE_COL, {})
    bio_effs = [results[c]["eff"] for c in BIO_COLS
                if c in results and results[c]["kind"] != "skip" and np.isfinite(results[c]["eff"])]
    bio_sig = [c for c in BIO_COLS if results.get(c, {}).get("holm", 1) is not None
               and (results.get(c, {}).get("holm", 1) < 0.05)]
    print("\n================= GATE VERDICT =================")
    print(f"  age effect |point-biserial| = {age_r.get('eff', float('nan')):.3f}"
          f"  (Holm p={age_r.get('holm', float('nan'))})  {age_r.get('note','')}")
    print(f"  age -> cluster 5-fold CV AUC = {auc:.3f}   (0.5 = age carries no split info)")
    print(f"  split aligns with PC{split_pc}; age lives strongest on PC{age_pc}"
          f"  (PC1 evr={evr[0]*100:.1f}%)")
    print(f"  biology (stage/N/M) max Cramér's V = "
          f"{max(bio_effs) if bio_effs else float('nan'):.3f}; Holm-sig: {bio_sig or 'none'}")
    print("  read: high age-AUC + split-axis==age-axis => age/era surrogate risk;")
    print("        biology also Holm-sig => the split carries disease signal beyond age.")
    print("================================================")
    print(f"\n[done] figures + CSVs in ./{OUT_DIR}/")
 
if __name__ == "__main__":
    main()