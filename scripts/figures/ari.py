import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

def load_embedding(path, EMB_ID_COL="patient_id"):
    df = pd.read_parquet(path)
    if EMB_ID_COL not in df.columns:
        raise ValueError(f"'{EMB_ID_COL}' not in parquet columns: {list(df.columns)[:6]}...")
    emb_cols = sorted([c for c in df.columns if c.startswith("c_")])
    if not emb_cols:
        emb_cols = sorted([c for c in df.columns if c != EMB_ID_COL and
                           pd.api.types.is_numeric_dtype(df[c])])
    X = df[emb_cols].to_numpy(dtype=np.float64)
    pid = df[EMB_ID_COL].astype(str).tolist()
    return X, pid, df

def reproduce_labels_k2(X):
    km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X)
    lab = km.labels_
    sizes = np.bincount(lab)
    small = int(np.argmin(sizes))
    canon = np.where(lab == small, 0, 1)
    return canon

path1 = r"c:\Users\boban\OneDrive\Desktop\BCI\MOGFormer\results\SSL\SURV_CLEAN\E600_p60_gated_meta_rna_meth\C_LumA.parquet"
path2 = r"c:\Users\boban\OneDrive\Desktop\BCI\MOGFormer\results\SSL\SURV_CLEAN\E600_p60_gated_meta_rna_meth\C_LumA_MOFA.parquet"

X1, pid1, df1 = load_embedding(path1)
X2, pid2, df2 = load_embedding(path2)

canon1 = reproduce_labels_k2(X1)
canon2 = reproduce_labels_k2(X2)

res1 = pd.DataFrame({"patient_id": pid1, "cluster1": canon1})
res2 = pd.DataFrame({"patient_id": pid2, "cluster2": canon2})

merged = pd.merge(res1, res2, on="patient_id", how="inner")
print(f"Number of overlapping patients: {len(merged)}")
ari = adjusted_rand_score(merged["cluster1"], merged["cluster2"])
print(f"Adjusted Rand Index (ARI): {ari}")

