import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from upsetplot import from_indicators
from upsetplot import plot as upset_plot
from lifelines import KaplanMeierFitter
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set paths
DATA_DIR = r"c:\Users\boban\OneDrive\Desktop\BCI\MOGFormer\data\clean"
OUT_DIR = r"c:\Users\boban\OneDrive\Desktop\BCI\MOGFormer\results\SSL\SURV_CLEAN"
os.makedirs(OUT_DIR, exist_ok=True)

# Set publication style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.title_fontsize': 14,
    'figure.titlesize': 18,
    'figure.titleweight': 'bold',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans']
})
sns.set_theme(style="ticks")

# Using colors extracted from the user's previous publications
subtype_palette = {
    "BRCA_LumA": "#6677a1",   # Muted Blue
    "BRCA_LumB": "#c4ae77",   # Tan / Gold
    "BRCA_Her2": "#4ca299",   # Teal
    "BRCA_Basal": "#90d490",  # Light Green
    "BRCA_Normal": "#262626", # Dark Gray
    "Unknown": "#b0b0b0"
}

def get_color(subtype):
    return subtype_palette.get(subtype, "#999999")

print("Loading datasets...")
surv_df = pd.read_csv(os.path.join(DATA_DIR, "survival_features_common.csv"))
surv_df['subtype'] = surv_df['subtype'].fillna('Unknown')
surv_df['subtype'] = pd.Categorical(surv_df['subtype'])

print("1. Generating Event Intersection Overview...")
try:
    events_df = surv_df[['OS', 'DSS', 'PFI', 'DFI']].fillna(0).astype(bool)
    events_df['Censored'] = ~events_df[['OS', 'DSS', 'PFI', 'DFI']].any(axis=1)
    upset_data = events_df.groupby(['OS', 'DSS', 'PFI', 'DFI', 'Censored']).size()
    
    fig = plt.figure(figsize=(10, 6))
    upset_plot(upset_data, fig=fig, element_size=45, show_counts='%d', facecolor=subtype_palette['BRCA_Basal'])
    plt.suptitle("Event Intersection Overview (All Patients)", fontsize=18, fontweight='bold', y=1.05)
    plt.savefig(os.path.join(OUT_DIR, "01_event_intersection_overview.png"), dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"Failed Plot 1: {e}")

print("2. Generating OS Event Rates by Subtype...")
os_rates = surv_df[surv_df['subtype'] != 'Unknown'].groupby('subtype', observed=True)['OS'].mean().reset_index()
plt.figure(figsize=(8, 6))
ax = sns.barplot(data=os_rates, x='subtype', y='OS', palette=subtype_palette, linewidth=0)
plt.title("Overall Survival (OS) Event Rate by Subtype", pad=20)
plt.ylabel("Event Rate Proportion")
plt.xlabel("PAM50 Subtype")
plt.xticks(rotation=45, ha='right')
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_os_event_rates_by_subtype.png"), dpi=300, bbox_inches='tight')
plt.close()

print("5. Generating Subtype Distribution...")
subtype_counts = surv_df[surv_df['subtype'] != 'Unknown']['subtype'].value_counts().reset_index()
plt.figure(figsize=(8, 6))
ax = sns.barplot(data=subtype_counts, y='subtype', x='count', palette=subtype_palette, linewidth=0)
plt.title("PAM50 Subtype Distribution in Cohort", pad=20)
plt.xlabel("Number of Patients")
plt.ylabel("Subtype")

total = len(surv_df[surv_df['subtype'] != 'Unknown'])
labels = [f"{v.get_width():.0f} ({(v.get_width()/total)*100:.1f}%)" for v in ax.containers[0]]
ax.bar_label(ax.containers[0], labels=labels, padding=5)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "05_subtype_distribution.png"), dpi=300, bbox_inches='tight')
plt.close()

print("6. Generating Age Distribution...")
plt.figure(figsize=(10, 6))
sns.kdeplot(data=surv_df[surv_df['subtype'] != 'Unknown'], x='clinical__demographic.age_at_index', 
            hue='subtype', palette=subtype_palette, fill=True, alpha=0.3, common_norm=False, linewidth=2)
plt.title("Density of Age at Index by Subtype", pad=20)
plt.xlabel("Age at Index (years)")
plt.ylabel("Density")
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "06_age_at_index_by_subtype.png"), dpi=300, bbox_inches='tight')
plt.close()

print("7 & 8. Generating Kaplan-Meier Curves...")
try:
    fig, ax = plt.subplots(figsize=(10, 7))
    kmf = KaplanMeierFitter()
    for name, grouped_df in surv_df.groupby('subtype', observed=True):
        valid = grouped_df.dropna(subset=['OS.time', 'OS'])
        if name != 'Unknown' and not valid.empty:
            kmf.fit(valid['OS.time'], valid['OS'], label=name)
            kmf.plot_survival_function(ax=ax, color=get_color(name), ci_show=False, linewidth=2.5)
    plt.title("Kaplan-Meier Survival Curve (Overall Survival)", pad=20)
    plt.ylabel("Survival Probability")
    plt.xlabel("Time (days)")
    ax.legend(title="Subtype", frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "07_km_survival_os_by_subtype.png"), dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 7))
    for name, grouped_df in surv_df.groupby('subtype', observed=True):
        valid = grouped_df.dropna(subset=['PFI.time', 'PFI'])
        if name != 'Unknown' and not valid.empty:
            kmf.fit(valid['PFI.time'], valid['PFI'], label=name)
            kmf.plot_survival_function(ax=ax, color=get_color(name), ci_show=False, linewidth=2.5)
    plt.title("Kaplan-Meier Survival Curve (Progression-Free Interval)", pad=20)
    plt.ylabel("Progression-Free Probability")
    plt.xlabel("Time (days)")
    ax.legend(title="Subtype", frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "08_km_survival_pfi_by_subtype.png"), dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"Failed KM curves: {e}")

print("9. Generating Stage Distribution...")
stage_col = 'clinical__diagnoses.ajcc_pathologic_stage'
if stage_col in surv_df.columns:
    valid_stage = surv_df[surv_df['subtype'] != 'Unknown'].copy()
    stage_dist = pd.crosstab(valid_stage['subtype'], valid_stage[stage_col], normalize='index')
    ax = stage_dist.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='viridis', edgecolor="white", linewidth=1)
    plt.title("AJCC Pathologic Stage Distribution by Subtype", pad=20)
    plt.ylabel("Proportion")
    plt.xlabel("PAM50 Subtype")
    plt.legend(title='Stage', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.xticks(rotation=45, ha='right')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "09_stage_distribution_by_subtype.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_tsne(df, feature_name, common_patients, surv_subset):
    try:
        transposed = df.T.loc[common_patients].fillna(0)
        scaled = StandardScaler().fit_transform(transposed)
        tsne_results = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(scaled)
        
        # All Subtypes
        tsne_df = pd.DataFrame({'t-SNE 1': tsne_results[:, 0], 't-SNE 2': tsne_results[:, 1], 'Subtype': surv_subset['subtype'].values, 'OS': surv_subset['OS'].values})
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=tsne_df[tsne_df['Subtype'] != 'Unknown'], x='t-SNE 1', y='t-SNE 2', hue='Subtype', palette=subtype_palette, alpha=0.8, s=60, linewidth=0)
        plt.title(f"t-SNE of {feature_name} (All Subtypes)", pad=20)
        plt.legend(frameon=False)
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"12_{feature_name.lower()}_tsne_all.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # LumA Only by OS
        luma_df = tsne_df[tsne_df['Subtype'] == 'BRCA_LumA']
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=luma_df, x='t-SNE 1', y='t-SNE 2', hue='OS', palette={0: "#3498db", 1: "#e74c3c"}, alpha=0.8, s=80, linewidth=0)
        plt.title(f"t-SNE of {feature_name} (Luminal A Only, by OS)", pad=20)
        plt.legend(title='OS Event', frameon=False, labels=["Censored (0)", "Event (1)"])
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"12_{feature_name.lower()}_tsne_luma_os.png"), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Failed {feature_name} t-SNE: {e}")

def plot_umap(df, feature_name, common_patients, surv_subset):
    try:
        transposed = df.T.loc[common_patients].fillna(0)
        scaled = StandardScaler().fit_transform(transposed)
        
        # Using UMAP
        reducer = umap.UMAP(n_components=2, random_state=42)
        umap_results = reducer.fit_transform(scaled)
        
        # All Subtypes
        umap_df = pd.DataFrame({'UMAP 1': umap_results[:, 0], 'UMAP 2': umap_results[:, 1], 'Subtype': surv_subset['subtype'].values, 'OS': surv_subset['OS'].values})
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=umap_df[umap_df['Subtype'] != 'Unknown'], x='UMAP 1', y='UMAP 2', hue='Subtype', palette=subtype_palette, alpha=0.8, s=60, linewidth=0)
        plt.title(f"UMAP of {feature_name} (All Subtypes)", pad=20)
        plt.legend(frameon=False)
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"12_{feature_name.lower()}_umap_all.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # LumA Only by OS
        luma_df = umap_df[umap_df['Subtype'] == 'BRCA_LumA']
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=luma_df, x='UMAP 1', y='UMAP 2', hue='OS', palette={0: "#3498db", 1: "#e74c3c"}, alpha=0.8, s=80, linewidth=0)
        plt.title(f"UMAP of {feature_name} (Luminal A Only, by OS)", pad=20)
        plt.legend(title='OS Event', frameon=False, labels=["Censored (0)", "Event (1)"])
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"12_{feature_name.lower()}_umap_luma_os.png"), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Failed {feature_name} UMAP: {e}")

print("12. Generating t-SNEs for Omics Data...")
rna_df = pd.read_csv(os.path.join(DATA_DIR, "data_rna_common.csv"), index_col=0)
cnv_df = pd.read_csv(os.path.join(DATA_DIR, "data_cnv_common.csv"), index_col=0)

# Need to check which methylation file to load, loading _common_meta.csv
meth_df = pd.read_csv(os.path.join(DATA_DIR, "gene_promoter_methylation_common_meta.csv"), index_col=0)

common_patients = rna_df.columns.intersection(surv_df['patient_id'])
surv_subset = surv_df.set_index('patient_id').loc[common_patients]

plot_tsne(rna_df, "RNA", common_patients, surv_subset)
plot_tsne(cnv_df, "CNV", common_patients, surv_subset)
plot_tsne(meth_df, "Methylation", common_patients, surv_subset)

print("Generating UMAPs for Omics Data...")
plot_umap(rna_df, "RNA", common_patients, surv_subset)
plot_umap(cnv_df, "CNV", common_patients, surv_subset)
plot_umap(meth_df, "Methylation", common_patients, surv_subset)

print("15. Generating Age vs OS Time Scatter (JointPlot)...")
valid_surv = surv_df[surv_df['subtype'] != 'Unknown']
g = sns.jointplot(data=valid_surv, x='clinical__demographic.age_at_index', y='OS.time', 
                  hue='subtype', palette=subtype_palette, alpha=0.7, height=8, s=50, linewidth=0)
g.fig.suptitle("Age vs Overall Survival Time with Marginal Distributions", y=1.03, fontweight='bold', fontsize=16)
g.set_axis_labels("Age at Index (years)", "Overall Survival Time (days)")
plt.savefig(os.path.join(OUT_DIR, "15_age_vs_ostime_jointplot.png"), dpi=300, bbox_inches='tight')
plt.close()

print("16. Generating LumA Event Intersection Overview...")
try:
    luma_df = surv_df[surv_df['subtype'] == 'BRCA_LumA']
    events_luma = luma_df[['OS', 'DSS', 'PFI', 'DFI']].fillna(0).astype(bool)
    events_luma['Censored'] = ~events_luma[['OS', 'DSS', 'PFI', 'DFI']].any(axis=1)
    upset_data_luma = events_luma.groupby(['OS', 'DSS', 'PFI', 'DFI', 'Censored']).size()
    
    fig = plt.figure(figsize=(10, 6))
    upset_plot(upset_data_luma, fig=fig, element_size=45, show_counts='%d', facecolor=subtype_palette['BRCA_LumA'])
    plt.suptitle("Event Intersection Overview (LumA Subtype)", fontsize=18, fontweight='bold', y=1.05)
    plt.savefig(os.path.join(OUT_DIR, "16_luma_event_intersection.png"), dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"Failed Plot 16: {e}")

print("Plots successfully generated based on new feedback.")
