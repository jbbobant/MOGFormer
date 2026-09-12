"""
Publication-grade visualisations for PPI and GRN subnetworks
derived from the MOGFormer selected-gene universe (435 genes).

Outputs (PNG 600 dpi + SVG) in  figures/ :
  PPI:
    1. ppi_degree_distribution   – degree histogram + log-log inset
    2. ppi_score_distribution    – combined_score histogram
    3. ppi_top_hubs              – horizontal bar of top-30 hub genes
    4. ppi_hop_distribution      – from earlier data (bar + cumulative)
  GRN:
    1. grn_degree_distribution   – in/out degree histograms
    2. grn_top_regulators        – top-25 TFs by out-degree
    3. grn_top_targets           – top-25 targets by in-degree
    4. grn_sign_balance          – activation vs repression per TF
    5. grn_network               – force-directed layout of the GRN
"""

import json, warnings, sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import networkx as nx

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════
ROOT    = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results/SSL/SURV_CLEAN/DATA/Graph"
OUT_DIR.mkdir(exist_ok=True)

GENE_JSON  = ROOT / "results" / "SSL" / "SURV_CLEAN" / "E600_p60_gated_meta" / "gene_universe.json"
PPI_ALIAS  = ROOT / "data" / "clean" / "9606.protein.aliases.v12.0.txt"
PPI_LINKS  = ROOT / "data" / "clean" / "9606.protein.links.v12.0.txt"
GRN_FILE   = ROOT / "data" / "clean" / "collectri_signed_edges.tsv"

# ═══════════════════════════════════════════════════
# GLOBAL STYLE
# ═══════════════════════════════════════════════════
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         11,
    "axes.titlesize":    14,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi":        150,
    "savefig.dpi":       600,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.15,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

# ── PPI palette  (teal / indigo spectrum) ─────────
PPI_PRIMARY   = "#0D6E6E"   # deep teal  (main bars, nodes)
PPI_DARK      = "#094B5A"   # navy-teal  (high-degree nodes)
PPI_MID       = "#2A9D8F"   # mid-teal
PPI_LIGHT     = "#76C7C0"   # light teal
PPI_ACCENT    = "#E76F51"   # terracotta accent (median lines)
PPI_ACCENT2   = "#E9C46A"   # gold accent (mean lines)
PPI_CMAP      = "GnBu"      # sequential colourmap for nodes

# ── GRN palette  (amber / sage / coral) ───────────
GRN_ACTIV     = "#588157"   # sage green  (activation)
GRN_REPRESS   = "#BC4749"   # brick red   (repression)
GRN_TF_ONLY   = "#386641"   # dark sage   (TF-only nodes)
GRN_DUAL      = "#DDA15E"   # amber       (dual-role nodes)
GRN_TARGET    = "#457B9D"   # steel blue  (target-only nodes)
GRN_CMAP      = "YlOrBr"    # sequential colourmap (unused)

# ── Shared ────────────────────────────────────────
C_GREY      = "#9E9E9E"
C_GREY_LT   = "#E0E0E0"
EDGE_CLR    = "#2B2B2B"

# Legacy aliases for non-network figures
C_BLUE     = PPI_PRIMARY
C_BLUE_MED = PPI_MID
C_BLUE_LT  = PPI_LIGHT
C_RED      = GRN_REPRESS
C_RED_LT   = "#E8A0A1"
C_GREEN    = GRN_ACTIV
C_GREEN_DK = GRN_TF_ONLY
C_ORANGE   = GRN_DUAL
C_PURPLE   = "#7B4F9E"


def save(fig, name):
    """Save figure as PNG + SVG."""
    fig.savefig(OUT_DIR / f"{name}.png")
    fig.savefig(OUT_DIR / f"{name}.svg")
    print(f"  -> {name}.png / .svg")
    plt.close(fig)


# ═══════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════
print("Loading data ...")

with open(GENE_JSON) as f:
    genes = set(json.load(f)["selected_genes"])
print(f"  Gene universe : {len(genes)} genes")

# --- PPI ---
print("  Loading PPI aliases ...")
aliases = pd.read_csv(PPI_ALIAS, sep="\t", comment="#", header=None,
                       names=["string_id", "alias", "source_db"])
gene_aliases = aliases[aliases["alias"].isin(genes)]
gene2string = gene_aliases.drop_duplicates("alias").set_index("alias")["string_id"].to_dict()
string2gene = {v: k for k, v in gene2string.items()}

print("  Loading PPI links (this takes a moment) ...")
ppi_raw = pd.read_csv(PPI_LINKS, sep=" ")
string_ids = set(gene2string.values())
ppi = ppi_raw[(ppi_raw["protein1"].isin(string_ids)) &
              (ppi_raw["protein2"].isin(string_ids))].copy()
ppi = ppi[ppi["combined_score"] >= 400].copy()
ppi["gene1"] = ppi["protein1"].map(string2gene)
ppi["gene2"] = ppi["protein2"].map(string2gene)
ppi = ppi.dropna(subset=["gene1", "gene2"])

G_ppi = nx.Graph()
for _, r in ppi.iterrows():
    G_ppi.add_edge(r["gene1"], r["gene2"], weight=r["combined_score"])
print(f"  PPI subgraph  : {G_ppi.number_of_nodes()} nodes, {G_ppi.number_of_edges()} edges")

# --- GRN ---
print("  Loading GRN ...")
grn = pd.read_csv(GRN_FILE, sep="\t")
grn_sub = grn[(grn["source"].isin(genes)) & (grn["target"].isin(genes))].copy()
G_grn = nx.DiGraph()
for _, r in grn_sub.iterrows():
    G_grn.add_edge(r["source"], r["target"], sign=int(r["sign"]))
print(f"  GRN subgraph  : {G_grn.number_of_nodes()} nodes, {G_grn.number_of_edges()} edges")
print()


# ╔══════════════════════════════════════════════════╗
# ║  PPI VISUALISATIONS                              ║
# ╚══════════════════════════════════════════════════╝
print("=== PPI FIGURES ===")

# ──────────────────────────────────────────
# PPI-1  Degree Distribution
# ──────────────────────────────────────────
degrees_ppi = np.array([d for _, d in G_ppi.degree()])

fig, ax = plt.subplots(figsize=(6.5, 4.2))
bins = np.arange(0, degrees_ppi.max() + 5, 5)
ax.hist(degrees_ppi, bins=bins, color=C_BLUE, edgecolor="white",
        linewidth=0.5, alpha=0.88, zorder=3)
ax.set_xlabel("Node degree")
ax.set_ylabel("Number of genes")
ax.set_title("PPI Network Degree Distribution", weight="bold", pad=10)
ax.axvline(np.median(degrees_ppi), color=C_RED, ls="--", lw=1.5, zorder=4,
           label=f"Median = {np.median(degrees_ppi):.0f}")
ax.axvline(np.mean(degrees_ppi), color=C_ORANGE, ls=":", lw=1.5, zorder=4,
           label=f"Mean = {np.mean(degrees_ppi):.1f}")
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(axis="y", ls="--", lw=0.4, alpha=0.5, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Log-log inset
axin = ax.inset_axes([0.55, 0.45, 0.42, 0.48])
deg_counts = Counter(degrees_ppi)
xs = sorted(deg_counts.keys())
ys = [deg_counts[x] for x in xs]
axin.scatter(xs, ys, s=14, c=C_BLUE, alpha=0.7, zorder=3, edgecolors="white", linewidths=0.3)
axin.set_xscale("log")
axin.set_yscale("log")
axin.set_xlabel("Degree (log)", fontsize=8)
axin.set_ylabel("Count (log)", fontsize=8)
axin.tick_params(labelsize=7)
axin.set_title("Log-log scale", fontsize=8, pad=3)
axin.grid(True, ls="--", lw=0.3, alpha=0.4)
for sp in axin.spines.values():
    sp.set_linewidth(0.5)

fig.tight_layout()
save(fig, "ppi_degree_distribution")


# ──────────────────────────────────────────
# PPI-2  Combined Score Distribution
# ──────────────────────────────────────────
scores = ppi["combined_score"].values

fig, ax = plt.subplots(figsize=(6.5, 4))
bins_s = np.arange(400, 1010, 25)
n, _, patches = ax.hist(scores, bins=bins_s, edgecolor="white", linewidth=0.4, zorder=3)

# Gradient colouring by score
norm = plt.Normalize(400, 1000)
cmap = plt.cm.YlOrRd
for p, left in zip(patches, bins_s[:-1]):
    p.set_facecolor(cmap(norm(left + 12)))

ax.set_xlabel("STRING Combined Score")
ax.set_ylabel("Number of interactions")
ax.set_title("PPI Interaction Confidence Distribution", weight="bold", pad=10)

# Threshold annotations
for thr, lab, clr in [(700, "High confidence", C_BLUE), (900, "Highest confidence", C_RED)]:
    cnt = (scores >= thr).sum()
    ax.axvline(thr, color=clr, ls="--", lw=1.4, zorder=4)
    ax.text(thr + 8, ax.get_ylim()[1] * 0.92, f"{lab}\n({cnt:,} edges)",
            fontsize=8, color=clr, va="top")

ax.grid(axis="y", ls="--", lw=0.4, alpha=0.5, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
save(fig, "ppi_score_distribution")


# ──────────────────────────────────────────
# PPI-3  Top-30 Hub Genes
# ──────────────────────────────────────────
deg_dict = dict(G_ppi.degree())
top30 = sorted(deg_dict.items(), key=lambda x: x[1], reverse=True)[:30]
hub_genes, hub_degs = zip(*top30)

fig, ax = plt.subplots(figsize=(7, 6))
y_pos = np.arange(len(hub_genes))[::-1]
colours = [cmap(norm_val) for norm_val, cmap_fn in
           [(d / max(hub_degs), plt.cm.Blues)] for d in hub_degs] if False else []
# Simple gradient
norm_d = plt.Normalize(min(hub_degs), max(hub_degs))
colours = [plt.cm.Blues(norm_d(d) * 0.7 + 0.3) for d in hub_degs]

bars = ax.barh(y_pos, hub_degs, height=0.68, color=colours,
               edgecolor=EDGE_CLR, linewidth=0.5, zorder=3)
ax.set_yticks(y_pos)
ax.set_yticklabels(hub_genes, fontsize=9)
ax.set_xlabel("Degree (number of PPI partners)")
ax.set_title("Top-30 PPI Hub Genes", weight="bold", pad=10)
ax.set_xlim(0, max(hub_degs) * 1.18)

for bar, deg in zip(bars, hub_degs):
    ax.text(bar.get_width() + max(hub_degs) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            str(deg), va="center", ha="left", fontsize=8.5, color="#333")

ax.grid(axis="x", ls="--", lw=0.4, alpha=0.5, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
save(fig, "ppi_top_hubs")


# ──────────────────────────────────────────
# PPI-4  Network Overview  (score ≥ 400, community layout)
# ──────────────────────────────────────────
print("  Computing PPI community layout (score ≥ 400) ...")

# Use the full ≥400 graph; keep largest connected component
if G_ppi.number_of_nodes() > 0:
    lcc_nodes = max(nx.connected_components(G_ppi), key=len)
    G_vis = G_ppi.subgraph(lcc_nodes).copy()
else:
    G_vis = G_ppi.copy()

print(f"  PPI (≥400, LCC): {G_vis.number_of_nodes()} nodes, "
      f"{G_vis.number_of_edges()} edges")

# ── Community detection (Louvain) ──
try:
    communities = nx.community.louvain_communities(G_vis, seed=42, resolution=1.0)
except AttributeError:
    communities = list(nx.community.greedy_modularity_communities(G_vis))
communities = sorted(communities, key=len, reverse=True)  # largest first
node2comm = {}
for ci, comm in enumerate(communities):
    for n in comm:
        node2comm[n] = ci
n_comm = len(communities)
print(f"  Detected {n_comm} communities")

# ── Community-aware spring layout ──
# Boost intra-community attraction so clusters separate
G_layout = G_vis.copy()
for u, v in G_layout.edges():
    if node2comm[u] == node2comm[v]:
        G_layout[u][v]["lw"] = 8.0
    else:
        G_layout[u][v]["lw"] = 0.1

pos = nx.spring_layout(G_layout, k=2.2 / np.sqrt(G_vis.number_of_nodes()),
                        iterations=200, seed=42, weight="lw")

fig, ax = plt.subplots(figsize=(13, 13))
ax.set_facecolor("#FAFAFA")
fig.patch.set_facecolor("#FAFAFA")

degs_vis = dict(G_vis.degree())
max_deg = max(degs_vis.values()) if degs_vis else 1

# ── Qualitative palette for communities ──
_comm_base = ["#2A9D8F", "#E76F51", "#264653", "#E9C46A",
              "#457B9D", "#F4A261", "#606C38", "#9B2226",
              "#6A4C93", "#1982C4", "#8AC926", "#FF595E"]
comm_colors = (_comm_base * ((n_comm // len(_comm_base)) + 1))[:n_comm]
node_colors = [comm_colors[node2comm[n]] for n in G_vis.nodes()]
node_sizes  = [max(20, 6 * np.sqrt(degs_vis[n] / max_deg) * 80) for n in G_vis.nodes()]

# ── Edges: intra-community very faint; inter-community even fainter ──
intra_edges = [(u, v) for u, v in G_vis.edges() if node2comm[u] == node2comm[v]]
inter_edges = [(u, v) for u, v in G_vis.edges() if node2comm[u] != node2comm[v]]
nx.draw_networkx_edges(G_vis, pos, edgelist=intra_edges, ax=ax,
                        alpha=0.04, width=0.20, edge_color="#444")
nx.draw_networkx_edges(G_vis, pos, edgelist=inter_edges, ax=ax,
                        alpha=0.015, width=0.12, edge_color="#999")

# ── Nodes ──
nx.draw_networkx_nodes(G_vis, pos, ax=ax,
                        node_size=node_sizes,
                        node_color=node_colors,
                        edgecolors="white", linewidths=0.5,
                        alpha=0.90)

# ── Labels: top-3 hubs per community (avoid overcrowding) ──
labels_to_draw = {}
for ci, comm in enumerate(communities):
    comm_degs = {n: degs_vis[n] for n in comm}
    top_k = min(3, len(comm_degs))
    for n, _ in sorted(comm_degs.items(), key=lambda x: x[1], reverse=True)[:top_k]:
        labels_to_draw[n] = n

label_bbox = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.82)
for node, label in labels_to_draw.items():
    x, y = pos[node]
    ax.text(x, y, label, fontsize=7, fontweight="bold", color="#1A1A1A",
            ha="center", va="center", bbox=label_bbox, zorder=5)

ax.set_title("PPI Network  (STRING score $\\geq$ 400, Largest Component)",
             weight="bold", fontsize=14, pad=14)
ax.axis("off")

# Stats box
stats_txt = (f"Nodes: {G_vis.number_of_nodes()}   "
             f"Edges: {G_vis.number_of_edges():,}\n"
             f"Communities: {n_comm}   "
             f"Med. degree: {np.median(list(degs_vis.values())):.0f}")
ax.text(0.02, 0.02, stats_txt, transform=ax.transAxes,
        fontsize=9, color="#555", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CCC", alpha=0.85))

# Community legend (top-6 largest by size)
comm_legend = []
for ci in range(min(6, n_comm)):
    comm_legend.append(
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=comm_colors[ci], markersize=9,
               label=f"Community {ci+1} ({len(communities[ci])})")
    )
ax.legend(handles=comm_legend, loc="upper left", fontsize=8.5,
          framealpha=0.92, edgecolor="#CCC", fancybox=True, title="Communities",
          title_fontsize=9)

fig.tight_layout()
save(fig, "ppi_network_overview")


# ╔══════════════════════════════════════════════════╗
# ║  GRN VISUALISATIONS                              ║
# ╚══════════════════════════════════════════════════╝
print("\n=== GRN FIGURES ===")

# ──────────────────────────────────────────
# GRN-1  In/Out Degree Distribution
# ──────────────────────────────────────────
in_deg  = np.array([d for _, d in G_grn.in_degree()])
out_deg = np.array([d for _, d in G_grn.out_degree()])

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

# Out-degree (TFs)
bins_out = np.arange(0, out_deg.max() + 3, 2)
ax_a.hist(out_deg[out_deg > 0], bins=bins_out, color=C_GREEN, edgecolor="white",
          linewidth=0.5, alpha=0.85, zorder=3)
ax_a.set_xlabel("Out-degree (# target genes)")
ax_a.set_ylabel("Number of TFs")
ax_a.set_title("A  TF Out-Degree", weight="bold", loc="left", fontsize=13)
ax_a.axvline(np.median(out_deg[out_deg > 0]), color=C_RED, ls="--", lw=1.3,
             label=f"Median = {np.median(out_deg[out_deg>0]):.0f}")
ax_a.legend(fontsize=9)
ax_a.grid(axis="y", ls="--", lw=0.4, alpha=0.5, zorder=0)
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)

# In-degree (targets)
bins_in = np.arange(0, in_deg.max() + 3, 2)
ax_b.hist(in_deg[in_deg > 0], bins=bins_in, color=C_PURPLE, edgecolor="white",
          linewidth=0.5, alpha=0.85, zorder=3)
ax_b.set_xlabel("In-degree (# regulators)")
ax_b.set_ylabel("Number of target genes")
ax_b.set_title("B  Target In-Degree", weight="bold", loc="left", fontsize=13)
ax_b.axvline(np.median(in_deg[in_deg > 0]), color=C_RED, ls="--", lw=1.3,
             label=f"Median = {np.median(in_deg[in_deg>0]):.0f}")
ax_b.legend(fontsize=9)
ax_b.grid(axis="y", ls="--", lw=0.4, alpha=0.5, zorder=0)
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)

fig.tight_layout()
save(fig, "grn_degree_distribution")


# ──────────────────────────────────────────
# GRN-2  Top-25 Transcription Factors (out-degree)
# ──────────────────────────────────────────
out_dict = dict(G_grn.out_degree())
top_tfs = sorted(out_dict.items(), key=lambda x: x[1], reverse=True)[:25]
tf_names, tf_outs = zip(*top_tfs)

# Count activating vs repressing per TF
tf_act = {}
tf_rep = {}
for tf in tf_names:
    signs = [G_grn[tf][t]["sign"] for t in G_grn.successors(tf)]
    tf_act[tf] = sum(1 for s in signs if s == 1)
    tf_rep[tf] = sum(1 for s in signs if s == -1)

fig, ax = plt.subplots(figsize=(7, 6.5))
y_pos = np.arange(len(tf_names))[::-1]

# Stacked bars: activation + repression
bars_act = ax.barh(y_pos, [tf_act[t] for t in tf_names], height=0.68,
                    color=C_GREEN, edgecolor="white", linewidth=0.5,
                    label="Activation", zorder=3)
bars_rep = ax.barh(y_pos, [tf_rep[t] for t in tf_names], height=0.68,
                    left=[tf_act[t] for t in tf_names],
                    color=C_RED, edgecolor="white", linewidth=0.5,
                    label="Repression", zorder=3)

ax.set_yticks(y_pos)
ax.set_yticklabels(tf_names, fontsize=9, style="italic")
ax.set_xlabel("Number of target genes")
ax.set_title("Top-25 Transcription Factors (GRN Out-Degree)", weight="bold", pad=10)
ax.legend(fontsize=9, loc="lower right", framealpha=0.9)

for i, tf in enumerate(tf_names):
    total = tf_act[tf] + tf_rep[tf]
    ax.text(total + max(tf_outs) * 0.015, y_pos[i],
            str(total), va="center", ha="left", fontsize=8.5, color="#333")

ax.set_xlim(0, max(tf_outs) * 1.15)
ax.grid(axis="x", ls="--", lw=0.4, alpha=0.5, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
save(fig, "grn_top_regulators")


# ──────────────────────────────────────────
# GRN-3  Top-25 Target Genes (in-degree)
# ──────────────────────────────────────────
in_dict = dict(G_grn.in_degree())
top_targs = sorted(in_dict.items(), key=lambda x: x[1], reverse=True)[:25]
tg_names, tg_ins = zip(*top_targs)

# Count activation vs repression per target
tg_act = {}
tg_rep = {}
for tg in tg_names:
    signs = [G_grn[s][tg]["sign"] for s in G_grn.predecessors(tg)]
    tg_act[tg] = sum(1 for s in signs if s == 1)
    tg_rep[tg] = sum(1 for s in signs if s == -1)

fig, ax = plt.subplots(figsize=(7, 6.5))
y_pos = np.arange(len(tg_names))[::-1]

bars_act = ax.barh(y_pos, [tg_act[t] for t in tg_names], height=0.68,
                    color=C_BLUE_MED, edgecolor="white", linewidth=0.5,
                    label="Activated by", zorder=3)
bars_rep = ax.barh(y_pos, [tg_rep[t] for t in tg_names], height=0.68,
                    left=[tg_act[t] for t in tg_names],
                    color=C_RED_LT, edgecolor="white", linewidth=0.5,
                    label="Repressed by", zorder=3)

ax.set_yticks(y_pos)
ax.set_yticklabels(tg_names, fontsize=9, style="italic")
ax.set_xlabel("Number of regulators")
ax.set_title("Top-25 Target Genes (GRN In-Degree)", weight="bold", pad=10)
ax.legend(fontsize=9, loc="lower right", framealpha=0.9)

for i, tg in enumerate(tg_names):
    total = tg_act[tg] + tg_rep[tg]
    ax.text(total + max(tg_ins) * 0.015, y_pos[i],
            str(total), va="center", ha="left", fontsize=8.5, color="#333")

ax.set_xlim(0, max(tg_ins) * 1.15)
ax.grid(axis="x", ls="--", lw=0.4, alpha=0.5, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
save(fig, "grn_top_targets")


# ──────────────────────────────────────────
# GRN-4  Global Activation/Repression Balance
# ──────────────────────────────────────────
n_act = sum(1 for _, _, d in G_grn.edges(data=True) if d["sign"] == 1)
n_rep = sum(1 for _, _, d in G_grn.edges(data=True) if d["sign"] == -1)

fig, ax = plt.subplots(figsize=(5, 5))
wedges, texts, autotexts = ax.pie(
    [n_act, n_rep],
    labels=["Activation", "Repression"],
    colors=[C_GREEN, C_RED],
    autopct=lambda p: f"{p:.1f}%\n({int(round(p*sum([n_act,n_rep])/100)):,})",
    pctdistance=0.65,
    startangle=90,
    counterclock=False,
    wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
    textprops=dict(fontsize=11, weight="semibold"),
)
for at in autotexts:
    at.set_fontsize(9)
    at.set_color("#333")

ax.text(0, 0.04, f"{n_act + n_rep}", ha="center", va="center",
        fontsize=24, weight="bold", color="#333")
ax.text(0, -0.12, "total edges", ha="center", va="center",
        fontsize=9, color="#777")

ax.set_title("GRN Regulatory Edge Composition", weight="bold", fontsize=13, pad=14)
fig.tight_layout()
save(fig, "grn_sign_balance")


# ──────────────────────────────────────────
# GRN-5  Network Visualisation (shell layout)
# ──────────────────────────────────────────
print("  Computing GRN shell layout ...")

# Classify nodes
out_d = dict(G_grn.out_degree())
in_d  = dict(G_grn.in_degree())
total_d = {n: out_d.get(n, 0) + in_d.get(n, 0) for n in G_grn.nodes()}

tf_only   = sorted([n for n in G_grn.nodes() if out_d[n] > 0 and in_d[n] == 0],
                    key=lambda n: out_d[n], reverse=True)
dual_role = sorted([n for n in G_grn.nodes() if out_d[n] > 0 and in_d[n] > 0],
                    key=lambda n: total_d[n], reverse=True)
target_only = sorted([n for n in G_grn.nodes() if out_d[n] == 0],
                      key=lambda n: in_d[n], reverse=True)

print(f"  Roles:  TF-only={len(tf_only)},  dual={len(dual_role)},  "
      f"target-only={len(target_only)}")

# Shell layout: TFs inner → dual middle → targets outer
shells = []
if tf_only:
    shells.append(tf_only)
if dual_role:
    shells.append(dual_role)
if target_only:
    shells.append(target_only)

pos_grn = nx.shell_layout(G_grn, nlist=shells, rotate=np.pi/2)

fig, ax = plt.subplots(figsize=(14, 14))
ax.set_facecolor("#FAFAFA")
fig.patch.set_facecolor("#FAFAFA")

# Separate activating and repressing edges
act_edges = [(u, v) for u, v, d in G_grn.edges(data=True) if d["sign"] == 1]
rep_edges = [(u, v) for u, v, d in G_grn.edges(data=True) if d["sign"] == -1]

nx.draw_networkx_edges(G_grn, pos_grn, edgelist=act_edges, ax=ax,
                        alpha=0.06, width=0.25, edge_color=GRN_ACTIV,
                        arrows=True, arrowsize=3, arrowstyle="-|>",
                        connectionstyle="arc3,rad=0.08")
nx.draw_networkx_edges(G_grn, pos_grn, edgelist=rep_edges, ax=ax,
                        alpha=0.10, width=0.30, edge_color=GRN_REPRESS,
                        arrows=True, arrowsize=3, arrowstyle="-|>",
                        connectionstyle="arc3,rad=0.08")

# Node colours & sizes by role
node_colors_grn = []
for n in G_grn.nodes():
    if n in set(dual_role):
        node_colors_grn.append(GRN_DUAL)
    elif n in set(tf_only):
        node_colors_grn.append(GRN_TF_ONLY)
    else:
        node_colors_grn.append(GRN_TARGET)

max_td = max(total_d.values()) if total_d else 1
node_sizes_grn = [max(30, 8 * np.sqrt(total_d[n] / max_td) * 120)
                  for n in G_grn.nodes()]

nx.draw_networkx_nodes(G_grn, pos_grn, ax=ax,
                        node_size=node_sizes_grn,
                        node_color=node_colors_grn,
                        edgecolors="white", linewidths=0.6,
                        alpha=0.90)

# Labels: all TFs + top-10 targets
labels_grn = {n: n for n in tf_only + dual_role}
for n in target_only[:10]:
    labels_grn[n] = n

label_bbox_grn = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.82)
for node, label in labels_grn.items():
    x, y = pos_grn[node]
    fs = 8 if node in set(tf_only + dual_role) else 6.5
    ax.text(x, y, label, fontsize=fs, fontweight="bold", fontstyle="italic",
            color="#1A1A1A", ha="center", va="center",
            bbox=label_bbox_grn, zorder=5)

ax.set_title("Gene Regulatory Network  (CollecTRI)",
             weight="bold", fontsize=15, pad=14)
ax.axis("off")

# Stats box
grn_stats = (f"Nodes: {G_grn.number_of_nodes()}   "
             f"Edges: {G_grn.number_of_edges()}  "
             f"(act {len(act_edges)}, rep {len(rep_edges)})\n"
             f"TFs: {len(tf_only)+len(dual_role)}  "
             f"(TF-only {len(tf_only)}, dual {len(dual_role)})")
ax.text(0.98, 0.02, grn_stats, transform=ax.transAxes,
        fontsize=9, color="#555", va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CCC", alpha=0.85))

# Legend
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=GRN_TF_ONLY,
           markersize=10, label="TF only (inner ring)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=GRN_DUAL,
           markersize=10, label="TF + target (middle ring)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=GRN_TARGET,
           markersize=10, label="Target only (outer ring)"),
    Line2D([0], [0], color=GRN_ACTIV, lw=2, alpha=0.5, label="Activation"),
    Line2D([0], [0], color=GRN_REPRESS, lw=2, alpha=0.5, label="Repression"),
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=9.5,
          framealpha=0.92, edgecolor="#CCC", fancybox=True)

fig.tight_layout()
save(fig, "grn_network_overview")


# ╔══════════════════════════════════════════════════╗
# ║  COMBINED SUMMARY PANEL                          ║
# ╚══════════════════════════════════════════════════╝
print("\n=== SUMMARY FIGURE ===")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Panel A — PPI degree
ax = axes[0, 0]
bins = np.arange(0, degrees_ppi.max() + 5, 5)
ax.hist(degrees_ppi, bins=bins, color=C_BLUE, edgecolor="white",
        linewidth=0.4, alpha=0.88, zorder=3)
ax.set_xlabel("PPI Degree")
ax.set_ylabel("Count")
ax.set_title("A  PPI Degree Distribution", weight="bold", loc="left", fontsize=12)
ax.axvline(np.median(degrees_ppi), color=C_RED, ls="--", lw=1.2,
           label=f"Median={np.median(degrees_ppi):.0f}")
ax.legend(fontsize=8)
ax.grid(axis="y", ls="--", lw=0.3, alpha=0.4, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Panel B — PPI score
ax = axes[0, 1]
bins_s = np.arange(400, 1010, 25)
n_b, _, patches_b = ax.hist(scores, bins=bins_s, edgecolor="white",
                             linewidth=0.3, zorder=3)
for p, left in zip(patches_b, bins_s[:-1]):
    p.set_facecolor(cmap(norm(left + 12)))
ax.set_xlabel("STRING Score")
ax.set_ylabel("Count")
ax.set_title("B  PPI Confidence Distribution", weight="bold", loc="left", fontsize=12)
ax.axvline(700, color=C_BLUE, ls="--", lw=1.2)
ax.axvline(900, color=C_RED, ls="--", lw=1.2)
ax.grid(axis="y", ls="--", lw=0.3, alpha=0.4, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Panel C — GRN out-degree (top 15)
ax = axes[1, 0]
top15_tfs = top_tfs[:15]
tf_n15, tf_o15 = zip(*top15_tfs)
y_p = np.arange(len(tf_n15))[::-1]
ax.barh(y_p, [tf_act[t] for t in tf_n15], height=0.65,
        color=C_GREEN, edgecolor="white", linewidth=0.4, label="Activation", zorder=3)
ax.barh(y_p, [tf_rep[t] for t in tf_n15], height=0.65,
        left=[tf_act[t] for t in tf_n15],
        color=C_RED, edgecolor="white", linewidth=0.4, label="Repression", zorder=3)
ax.set_yticks(y_p)
ax.set_yticklabels(tf_n15, fontsize=8.5, style="italic")
ax.set_xlabel("# Targets")
ax.set_title("C  Top-15 Transcription Factors", weight="bold", loc="left", fontsize=12)
ax.legend(fontsize=8, loc="lower right")
ax.grid(axis="x", ls="--", lw=0.3, alpha=0.4, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Panel D — GRN sign balance
ax = axes[1, 1]
wedges, texts, autotexts = ax.pie(
    [n_act, n_rep],
    labels=["Activation", "Repression"],
    colors=[C_GREEN, C_RED],
    autopct=lambda p: f"{p:.1f}%",
    pctdistance=0.72,
    startangle=90, counterclock=False,
    wedgeprops=dict(width=0.40, edgecolor="white", linewidth=2),
    textprops=dict(fontsize=10),
)
for at in autotexts:
    at.set_fontsize(9)
    at.set_color("#333")
ax.text(0, 0.03, f"{n_act+n_rep}", ha="center", va="center",
        fontsize=18, weight="bold", color="#333")
ax.text(0, -0.10, "edges", ha="center", va="center", fontsize=8.5, color="#777")
ax.set_title("D  GRN Edge Composition", weight="bold", loc="left", fontsize=12)

fig.suptitle("Network Topology Summary — PPI & GRN",
             fontsize=15, weight="bold", y=1.01)
fig.tight_layout()
save(fig, "network_summary_panel")


print(f"\nDone! All figures in: {OUT_DIR}")
