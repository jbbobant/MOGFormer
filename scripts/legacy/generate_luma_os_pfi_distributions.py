import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from matplotlib_venn import venn2

def generate_plot():
    DATA_DIR = r"c:\Users\boban\OneDrive\Desktop\BCI\MOGFormer\data\clean"
    OUT_DIR = r"c:\Users\boban\OneDrive\Desktop\BCI\MOGFormer\results\SSL\SURV_CLEAN"
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. Setup style
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans']
    })
    sns.set_theme(style="white")
    
    # 2. Load data
    df = pd.read_csv(os.path.join(DATA_DIR, "survival_features_common.csv"))
    luma_df = df[df['subtype'] == 'BRCA_LumA'].copy()
    
    n_patients = len(luma_df)
    
    # Define colors
    c_event = '#90d490' # Light Green
    c_censored = '#6677a1' # Muted Blue
    c_intersection = '#4ca299' # Dark Teal / intersection mix
    
    # 3. Create Grid
    fig = plt.figure(figsize=(14, 20), dpi=300)
    fig.suptitle(f"BRCA LumA - OS & PFI Distributions\n(TCGA Pancan Atlas 2018) n={n_patients}", 
                 fontsize=18, fontweight='bold', y=0.98)
    
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(4, 2, height_ratios=[1, 1, 1, 1], figure=fig, hspace=0.4, wspace=0.3)
    
    # ---------------------------------------------------------
    # ROW 1: Time Distributions (colored by event status)
    # ---------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(data=luma_df, x='OS.time', hue='OS', palette={0: c_censored, 1: c_event},
                 kde=True, ax=ax1, alpha=0.6, linewidth=1, edgecolor='white')
    ax1.set_title("Overall Survival - Time Distribution")
    ax1.set_xlabel("Months")
    # Legend formatting
    handles = ax1.get_legend().legend_handles if ax1.get_legend() else []
    ax1.legend(title="", labels=['Event', 'Censored'])
    
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(data=luma_df, x='PFI.time', hue='PFI', palette={0: c_censored, 1: c_event},
                 kde=True, ax=ax2, alpha=0.6, linewidth=1, edgecolor='white')
    ax2.set_title("Progression-Free Interval - Time Distribution")
    ax2.set_xlabel("Months")
    ax2.legend(title="", labels=['Event', 'Censored'])
    
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel("Number of Patients")
        ax.yaxis.grid(True, linestyle='--', color='lightgrey', alpha=0.5)

    # ---------------------------------------------------------
    # ROW 2: Kaplan-Meier Curves
    # ---------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    kmf_os = KaplanMeierFitter()
    # Filter NaNs for OS
    os_data = luma_df.dropna(subset=['OS.time', 'OS'])
    kmf_os.fit(os_data['OS.time'], event_observed=os_data['OS'], label='OS')
    kmf_os.plot_survival_function(ax=ax3, color=c_event, linewidth=2.5, show_censors=True,
                                  censor_styles={'marker': '+', 'mfc': c_censored, 'mec': c_censored, 'ms': 8})
    ax3.set_title("OS - Kaplan-Meier Curve")
    ax3.set_xlabel("Months")
    ax3.set_ylabel("Survival Probability")
    median_os = kmf_os.median_survival_time_
    if not np.isinf(median_os):
        ax3.text(0.6, 0.8, f"Median = {median_os:.1f} mo", transform=ax3.transAxes, fontsize=12)
    
    ax4 = fig.add_subplot(gs[1, 1])
    kmf_pfi = KaplanMeierFitter()
    pfi_data = luma_df.dropna(subset=['PFI.time', 'PFI'])
    kmf_pfi.fit(pfi_data['PFI.time'], event_observed=pfi_data['PFI'], label='PFI')
    kmf_pfi.plot_survival_function(ax=ax4, color=c_event, linewidth=2.5, show_censors=True,
                                   censor_styles={'marker': '+', 'mfc': c_censored, 'mec': c_censored, 'ms': 8})
    ax4.set_title("PFI - Kaplan-Meier Curve")
    ax4.set_xlabel("Months")
    ax4.set_ylabel("Progression-Free Probability")
    median_pfi = kmf_pfi.median_survival_time_
    if not np.isinf(median_pfi):
        ax4.text(0.6, 0.8, f"Median = {median_pfi:.1f} mo", transform=ax4.transAxes, fontsize=12)

    for ax in [ax3, ax4]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, 1.05)
        ax.get_legend().remove()
        ax.yaxis.grid(True, linestyle='--', color='lightgrey', alpha=0.5)

    # ---------------------------------------------------------
    # ROW 3: Time by Event Status (Boxplot)
    # ---------------------------------------------------------
    ax5 = fig.add_subplot(gs[2, 0])
    sns.boxplot(data=os_data, x='OS', y='OS.time', hue='OS', palette=[c_censored, c_event], legend=False, ax=ax5, width=0.5, boxprops=dict(alpha=0.7))
    sns.stripplot(data=os_data, x='OS', y='OS.time', color='black', alpha=0.3, size=3, ax=ax5, jitter=True)
    ax5.set_title("OS - Time by Event Status")
    ax5.set_xticklabels(['Censored\n(OS=0)', 'Event\n(OS=1)'])
    ax5.set_xlabel("")
    ax5.set_ylabel("Months")
    
    # Annotate counts
    os_counts = os_data['OS'].value_counts()
    ax5.text(0, ax5.get_ylim()[1]*0.95, f"n={os_counts.get(0,0)}", ha='center', va='top', fontweight='bold', color=c_censored)
    ax5.text(1, ax5.get_ylim()[1]*0.95, f"n={os_counts.get(1,0)}", ha='center', va='top', fontweight='bold', color=c_event)

    ax6 = fig.add_subplot(gs[2, 1])
    sns.boxplot(data=pfi_data, x='PFI', y='PFI.time', hue='PFI', palette=[c_censored, c_event], legend=False, ax=ax6, width=0.5, boxprops=dict(alpha=0.7))
    sns.stripplot(data=pfi_data, x='PFI', y='PFI.time', color='black', alpha=0.3, size=3, ax=ax6, jitter=True)
    ax6.set_title("PFI - Time by Event Status")
    ax6.set_xticklabels(['Censored\n(PFI=0)', 'Event\n(PFI=1)'])
    ax6.set_xlabel("")
    ax6.set_ylabel("Months")
    
    pfi_counts = pfi_data['PFI'].value_counts()
    ax6.text(0, ax6.get_ylim()[1]*0.95, f"n={pfi_counts.get(0,0)}", ha='center', va='top', fontweight='bold', color=c_censored)
    ax6.text(1, ax6.get_ylim()[1]*0.95, f"n={pfi_counts.get(1,0)}", ha='center', va='top', fontweight='bold', color=c_event)

    for ax in [ax5, ax6]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', color='lightgrey', alpha=0.5)

    # ---------------------------------------------------------
    # ROW 4: Venn Diagram Intersection
    # ---------------------------------------------------------
    ax7 = fig.add_subplot(gs[3, :])
    ax7.set_title(f"OS x PFI Event Intersection (n={n_patients})", fontsize=16, fontweight='bold', pad=20)
    
    both_df = luma_df.dropna(subset=['OS', 'PFI'])
    set_os = set(both_df[both_df['OS'] == 1].index)
    set_pfi = set(both_df[both_df['PFI'] == 1].index)
    
    n_no_event = len(both_df) - len(set_os.union(set_pfi))
    pct_no_event = (n_no_event / len(both_df)) * 100
    
    v = venn2([set_os, set_pfi], set_labels=('OS Event', 'PFI Event'), ax=ax7, set_colors=(c_event, c_censored), alpha=0.7)
    
    if v.get_label_by_id('10'):
        v.get_label_by_id('10').set_fontsize(14)
    if v.get_label_by_id('01'):
        v.get_label_by_id('01').set_fontsize(14)
    if v.get_label_by_id('11'):
        v.get_label_by_id('11').set_fontsize(14)
        
    ax7.text(0.5, -0.1, f"Patients with No Events (Censored for both): {n_no_event} ({pct_no_event:.1f}%)", 
             ha='center', va='center', transform=ax7.transAxes, fontsize=14, fontweight='bold', color='grey')

    output_path = os.path.join(OUT_DIR, "18_BRCA_LumA_os_pfi_distributions.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    generate_plot()
