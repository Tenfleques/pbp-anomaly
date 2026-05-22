#!/usr/bin/env python3
"""
Generate all figures for the Nature environmental anomaly paper.

Reads from results/precomputed/ (or --results-dir) and outputs to figures/.

Figures:
  Fig 1: Temporal robustness (AQ + weather two-panel bar chart)
  Fig 2: Hub variable structure (AQ heatmap + weather pair ranking)
  Fig 3: Sensor-pair z-score case study (UCI interpretability)

Usage:
    python experiments/generate_figures.py
    python experiments/generate_figures.py --results-dir results/local
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_RESULTS = os.path.join(REPO_DIR, 'results', 'precomputed')
FIG_DIR = os.path.join(REPO_DIR, 'figures')

# Nature journal style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLOR_STD = '#2166ac'
COLOR_PAIR = '#72b2d7'
COLOR_VIS_HUB = '#2196F3'
COLOR_OTHER_PAIR = '#BDBDBD'

AQ_KEYS = ['UCI', 'Beijing', 'EPA_LA', 'SaoPaulo', 'CapeTown']
AQ_SHORT = {
    'UCI': 'UCI (Italy)', 'Beijing': 'Beijing', 'EPA_LA': 'EPA (LA)',
    'SaoPaulo': 'S\u00e3o Paulo', 'CapeTown': 'Cape Town',
}

STATION_ORDER = ['ORD', 'MIA', 'SFO', 'FAI']
STATION_LABELS = {
    'ORD': 'Chicago\n(ORD)', 'MIA': 'Miami\n(MIA)',
    'SFO': 'San Francisco\n(SFO)', 'FAI': 'Fairbanks\n(FAI)',
}

SHORT_NAMES = {
    'UCI Air Quality (Italy)': 'UCI (Italy)',
    'Beijing Multi-Site (Dongsi)': 'Beijing',
    'EPA AQS (LA, 2023)': 'EPA (LA)',
    'S\u00e3o Paulo (CAMS)': 'S\u00e3o Paulo',
    'Cape Town (CAMS)': 'Cape Town',
}

SENSOR_SHORT = {
    'PT08.S1(CO)': 'CO', 'PT08.S2(NMHC)': 'NMHC', 'PT08.S3(NOx)': 'NOx',
    'PT08.S4(NO2)': 'NO2', 'PT08.S5(O3)': 'O3', 'T': 'T', 'RH': 'RH', 'AH': 'AH',
}


def load_json(results_dir, filename):
    path = os.path.join(results_dir, filename)
    if not os.path.exists(path):
        print(f"  SKIP: {filename} not found")
        return None
    with open(path) as f:
        return json.load(f)


# ---- Figure 1: Temporal Robustness (two-panel) ----

def generate_figure1(transfer, weather):
    fig = plt.figure(figsize=(7.2, 3.5))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[5, 4], wspace=0.35)
    ax_aq = fig.add_subplot(gs[0, 0])
    ax_wx = fig.add_subplot(gs[0, 1])

    # Left panel: AQ datasets
    names, std_full, std_ts, pair_full, pair_ts = [], [], [], [], []
    for key in AQ_KEYS:
        r = transfer.get(key, {})
        if 'error' in r:
            continue
        ws6 = r.get('window_sweep_one_tailed', {}).get('6', {})
        ts = r.get('temporal_split', {})
        if 'error' in ws6 or 'error' in ts:
            continue
        names.append(AQ_SHORT.get(key, key))
        std_full.append(ws6['pbp_std']['auc_roc'])
        pair_full.append(ws6['pbp_pair']['auc_roc'])
        std_ts.append(ts.get('pbp_std', {}).get('auc_roc', 0))
        pair_ts.append(ts.get('pbp_pair', {}).get('auc_roc', 0))

    x = np.arange(len(names))
    width = 0.18
    ax_aq.bar(x - 1.5*width, std_full, width, color=COLOR_STD, label='PBP std (full)', alpha=0.9)
    ax_aq.bar(x - 0.5*width, std_ts, width, color=COLOR_STD, label='PBP std (temporal)', alpha=0.5, hatch='//')
    ax_aq.bar(x + 0.5*width, pair_full, width, color=COLOR_PAIR, label='PBP pair (full)', alpha=0.9)
    ax_aq.bar(x + 1.5*width, pair_ts, width, color=COLOR_PAIR, label='PBP pair (temporal)', alpha=0.5, hatch='//')

    ax_aq.set_ylabel('AUC-ROC')
    ax_aq.set_xticks(x)
    ax_aq.set_xticklabels(names, rotation=25, ha='right')
    ax_aq.set_ylim(0.5, 1.05)
    ax_aq.legend(ncol=2, fontsize=5.5, loc='lower left')
    ax_aq.set_title('a  Air quality datasets', fontweight='bold', loc='left')

    # Right panel: Weather stations (two-tailed)
    wx_names, wx_std_full, wx_std_ts, wx_pair_full, wx_pair_ts = [], [], [], [], []
    for code in STATION_ORDER:
        d = weather[code]
        two = d['two_tailed']
        ts = d['temporal_split']
        wx_names.append(STATION_LABELS[code])
        wx_std_full.append(two['pbp_std']['auc_roc'])
        wx_pair_full.append(two['pbp_pair']['auc_roc'])
        wx_std_ts.append(ts['pbp_std']['auc_roc'])
        wx_pair_ts.append(ts['pbp_pair']['auc_roc'])

    x2 = np.arange(len(wx_names))
    ax_wx.bar(x2 - 1.5*width, wx_std_full, width, color=COLOR_STD, alpha=0.9)
    ax_wx.bar(x2 - 0.5*width, wx_std_ts, width, color=COLOR_STD, alpha=0.5, hatch='//')
    ax_wx.bar(x2 + 0.5*width, wx_pair_full, width, color=COLOR_PAIR, alpha=0.9)
    ax_wx.bar(x2 + 1.5*width, wx_pair_ts, width, color=COLOR_PAIR, alpha=0.5, hatch='//')

    ax_wx.set_xticks(x2)
    ax_wx.set_xticklabels(wx_names, rotation=0, ha='center')
    ax_wx.set_ylim(0.5, 1.05)
    ax_wx.set_title('b  Weather stations (two-tailed)', fontweight='bold', loc='left')

    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig1_temporal_robustness')
    fig.savefig(out + '.pdf')
    fig.savefig(out + '.png')
    plt.close(fig)
    print(f"  Saved {out}.pdf/.png")


# ---- Figure 2: Hub Variable Structure (two-panel) ----

def generate_figure2(cross_env, weather, results_dir):
    fig = plt.figure(figsize=(7.2, 5.5))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[5, 4], wspace=0.45)
    ax_heat = fig.add_subplot(gs[0, 0])

    # Left panel: AQ heatmap
    csv_path = os.path.join(results_dir, 'cross_env_pair_analysis.csv')
    if not os.path.exists(csv_path):
        print("  SKIP figure 2: cross_env_pair_analysis.csv not found")
        plt.close(fig)
        return

    df = pd.read_csv(csv_path)
    df['pair'] = df['sensor_1'] + '\u2013' + df['sensor_2']
    pivot = df.pivot_table(index='pair', columns='dataset', values='auc')
    pivot = pivot.rename(columns=SHORT_NAMES)

    pivot['mean'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('mean', ascending=True)
    pivot = pivot.drop(columns='mean')

    im = ax_heat.imshow(pivot.values, aspect='auto', cmap='YlOrRd', vmin=0.5, vmax=1.0)
    ax_heat.set_xticks(range(len(pivot.columns)))
    ax_heat.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax_heat.set_yticks(range(len(pivot.index)))
    ax_heat.set_yticklabels(pivot.index, fontsize=6)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isnan(val):
                ax_heat.text(j, i, '\u2014', ha='center', va='center', fontsize=5, color='gray')
            else:
                color = 'white' if val > 0.85 else 'black'
                ax_heat.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=5, color=color)

    cbar = plt.colorbar(im, ax=ax_heat, shrink=0.6, label='AUC-ROC', pad=0.02)
    cbar.ax.tick_params(labelsize=6)

    universal_pairs = []
    for u in cross_env.get('universal_pairs', []):
        p = f"{u['pair'][0]}\u2013{u['pair'][1]}"
        universal_pairs.append(p)
    for i, pair_name in enumerate(pivot.index):
        if pair_name in universal_pairs:
            ax_heat.text(-0.6, i, '*', ha='center', va='center', fontsize=9,
                         color=COLOR_STD, fontweight='bold')

    ax_heat.set_title('a  Sensor-pair AUC across environments', fontweight='bold', loc='left')

    # Right panel: Weather pair ranking
    ax_placeholder = fig.add_subplot(gs[0, 1])
    ax_placeholder.set_visible(False)

    station_data = {}
    for code in STATION_ORDER:
        station_data[code] = weather[code]['two_tailed']['pair_ranking'][:8]

    gs_right = gs[0, 1].subgridspec(4, 1, hspace=0.5)
    for row_idx, code in enumerate(STATION_ORDER):
        ax_sub = fig.add_subplot(gs_right[row_idx, 0])
        top8 = station_data[code]
        names = [p['pair'] for p in top8]
        aucs = [p['auc'] for p in top8]
        colors = [COLOR_VIS_HUB if 'Vis' in n else COLOR_OTHER_PAIR for n in names]

        bars = ax_sub.barh(range(len(names)), aucs, color=colors, edgecolor='black', linewidth=0.3)
        ax_sub.set_yticks(range(len(names)))
        ax_sub.set_yticklabels(names, fontsize=5.5)
        ax_sub.invert_yaxis()
        ax_sub.set_xlim(0.55, 0.85)
        ax_sub.tick_params(axis='x', labelsize=6)

        for bar, val in zip(bars, aucs):
            ax_sub.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', va='center', fontsize=5)

        climate = weather[code].get('climate', '')
        label = f"{code} ({climate})" if climate else code
        if row_idx == 0:
            ax_sub.set_title('b  Weather pair ranking (visibility hub)', fontweight='bold',
                             loc='left', fontsize=9)
        ax_sub.text(0.98, 0.5, label, transform=ax_sub.transAxes, ha='right', va='center',
                    fontsize=7, fontstyle='italic', color='#555555')
        if row_idx == len(STATION_ORDER) - 1:
            ax_sub.set_xlabel('AUC-ROC', fontsize=7)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig2_pair_heatmap')
    fig.savefig(out + '.pdf')
    fig.savefig(out + '.png')
    plt.close(fig)
    print(f"  Saved {out}.pdf/.png")


# ---- Figure 3: Sensor-pair z-score case study ----

def generate_figure3(results_dir):
    path = os.path.join(results_dir, 'interpretability_case_study_results.json')
    if not os.path.exists(path):
        print("  SKIP figure 3: interpretability_case_study_results.json not found")
        return

    with open(path) as f:
        data = json.load(f)

    window = data['window_analyses'][0]
    pairs = window['pair_scores_ranked']

    labels = ['\u00d7'.join(SENSOR_SHORT.get(s, s) for s in p['pair']) for p in pairs]
    zscores = np.array([p['z_score'] for p in pairs])

    colors = []
    for z in zscores:
        if z > 3:
            colors.append('#d62728')
        elif z > 2:
            colors.append('#ff7f0e')
        elif z > 1:
            colors.append('#1f77b4')
        else:
            colors.append('#999999')

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(labels))
    ax.bar(x, zscores, color=colors, edgecolor='none', width=0.7)
    ax.axhline(y=2, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.text(len(labels) - 0.5, 2.1, 'z = 2', ha='right', va='bottom', fontsize=8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('z-score (vs. training baseline)')
    ax.set_xlim(-0.6, len(labels) - 0.4)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig3_pair_zscore_case_study')
    fig.savefig(out + '.pdf', bbox_inches='tight')
    fig.savefig(out + '.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Saved {out}.pdf/.png")


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description='Generate Nature paper figures')
    parser.add_argument('--results-dir', default=DEFAULT_RESULTS,
                        help='Directory containing result JSON/CSV files')
    args = parser.parse_args()
    results_dir = args.results_dir

    os.makedirs(FIG_DIR, exist_ok=True)

    print("Generating figures from:", results_dir)
    print("Output directory:", FIG_DIR)
    print("=" * 60)

    transfer = load_json(results_dir, 'aq_technique_transfer_results.json')
    weather = load_json(results_dir, 'weather_multi_station_results.json')
    cross_env = load_json(results_dir, 'cross_env_pair_analysis.json')

    if transfer and weather:
        print("\nFigure 1: Temporal robustness (two-panel)")
        generate_figure1(transfer, weather)
    else:
        print("\nFigure 1: SKIP (need aq_technique_transfer + weather results)")

    if cross_env and weather:
        print("\nFigure 2: Hub variable structure (two-panel)")
        generate_figure2(cross_env, weather, results_dir)
    else:
        print("\nFigure 2: SKIP (need cross_env_pair_analysis + weather results)")

    print("\nFigure 3: Sensor-pair z-score case study")
    generate_figure3(results_dir)

    print("\n" + "=" * 60)
    print("Done. Check figures/ directory.")


if __name__ == '__main__':
    main()
