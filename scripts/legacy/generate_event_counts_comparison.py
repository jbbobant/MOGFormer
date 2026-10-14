import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_plot():
    DATA_DIR = r"c:\Users\boban\OneDrive\Desktop\BCI\MOGFormer\data\clean"
    OUT_DIR = r"c:\Users\boban\OneDrive\Desktop\BCI\MOGFormer\results\SSL\SURV_CLEAN"
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 2. Read clinical data
    df = pd.read_csv(os.path.join(DATA_DIR, "survival_features_common.csv"))
    
    # Define endpoints
    endpoints = ['PFI', 'OS', 'DSS']
    titles = ['PFI', 'OS', 'DSS']
    
    # 4. Prepare data for plotting
    plot_data = []
    
    for ep in endpoints:
        # Time column is ep + '.time', event is ep
        event_col = ep
        # Drop NaNs for this endpoint
        valid_df = df.dropna(subset=[event_col])
        
        # Whole cohort
        n_whole = len(valid_df)
        events_whole = valid_df[event_col].sum()
        censored_whole = n_whole - events_whole
        
        # Luminal A only
        lumA_df = valid_df[valid_df['subtype'] == 'BRCA_LumA']
        n_lumA = len(lumA_df)
        events_lumA = lumA_df[event_col].sum()
        censored_lumA = n_lumA - events_lumA
        
        plot_data.append({
            'ep': ep,
            'whole': {'n': n_whole, 'events': events_whole, 'censored': censored_whole},
            'lumA': {'n': n_lumA, 'events': events_lumA, 'censored': censored_lumA}
        })
    
    # 5. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(12, 8), dpi=300)
    
    # Colors
    # User asked: "except for the red that must be changed to a light green like the other figures"
    # Original Red: '#c0392b' (Event Whole), '#df7a70' (Event LumA)
    # We replace red with light green (#90d490).
    # Whole event -> darker green to match contrast
    # LumA event -> the exact #90d490
    color_event_whole = '#509e50'
    color_event_lumA = '#90d490'
    color_censored_whole = '#2874a6'
    color_censored_lumA = '#85c1e9'
    
    total_n = len(df)
    
    fig.suptitle(f'Survival Event Counts — TCGA BRCA (n={total_n})\nWhole Cohort vs. Luminal A', fontsize=16, fontweight='bold')
    
    for ax, data, title in zip(axes, plot_data, titles):
        # Data
        whole = data['whole']
        lumA = data['lumA']
        
        events = [whole['events'], lumA['events']]
        censored = [whole['censored'], lumA['censored']]
        n_vals = [whole['n'], lumA['n']]
        
        # Plot stacked bars
        # Whole cohort
        bar_whole_event = ax.bar(0, events[0], color=color_event_whole, width=0.6, edgecolor='white')
        bar_whole_censored = ax.bar(0, censored[0], bottom=events[0], color=color_censored_whole, width=0.6, edgecolor='white')
        
        # Luminal A
        bar_lumA_event = ax.bar(1, events[1], color=color_event_lumA, width=0.6, edgecolor='white')
        bar_lumA_censored = ax.bar(1, censored[1], bottom=events[1], color=color_censored_lumA, width=0.6, edgecolor='white')
        
        # Formatting
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_ylabel('Number of patients', fontsize=12)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Whole\ncohort', 'Luminal A\nonly'], fontsize=12)
        
        # Spines and ticks
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=12)
        
        # Add grid lines
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax.set_axisbelow(True)
        
        # Limits
        ax.set_ylim(0, max(n_vals) + max(n_vals)*0.1)
        ax.set_xlim(-0.5, 1.5)
        
        # Add text for "n=..."
        ax.text(0, n_vals[0] + max(n_vals)*0.02, f"n={int(n_vals[0])}", ha='center', va='bottom', fontsize=10)
        ax.text(1, n_vals[1] + max(n_vals)*0.02, f"n={int(n_vals[1])}", ha='center', va='bottom', fontsize=10)
        
        # Add text inside bars
        # Whole events
        pct_whole = (events[0] / n_vals[0]) * 100
        ax.text(0, events[0] / 2, f"{int(events[0])}\n({pct_whole:.1f}%)", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        
        # Whole censored
        ax.text(0, events[0] + censored[0]/2, f"{int(censored[0])}", ha='center', va='center', color='white', fontsize=10)
        
        # Luminal A events
        pct_lumA = (events[1] / n_vals[1]) * 100
        ax.text(1, events[1] / 2, f"{int(events[1])}\n({pct_lumA:.1f}%)", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        
        # Luminal A censored
        ax.text(1, events[1] + censored[1]/2, f"{int(censored[1])}", ha='center', va='center', color='white', fontsize=10)
        
        # X-axis label
        if title == 'PFI':
            ax.set_xlabel('Progression-Free Interval', fontsize=10, color='dimgrey')
        elif title == 'OS':
            ax.set_xlabel('Overall Survival', fontsize=10, color='dimgrey')
        elif title == 'DSS':
            ax.set_xlabel('Disease-Specific Survival', fontsize=10, color='dimgrey')
            
    # Add common text at bottom
    fig.text(0.5, 0.08, "Event bars show count and event rate (%). Censored bars show number of patients without event.", ha='center', fontsize=10, color='grey', style='italic')
    
    # Legend
    import matplotlib.patches as mpatches
    patch1 = mpatches.Patch(color=color_event_whole, label='Event (Whole cohort)')
    patch2 = mpatches.Patch(color=color_event_lumA, label='Event (Luminal A)')
    patch3 = mpatches.Patch(color=color_censored_whole, label='Censored (Whole cohort)')
    patch4 = mpatches.Patch(color=color_censored_lumA, label='Censored (Luminal A)')
    
    fig.legend(handles=[patch1, patch3, patch2, patch4], loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=11, frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18, top=0.85)
    
    output_path = os.path.join(OUT_DIR, "17_event_counts_cohort_vs_luma.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    generate_plot()
