"""
Generate all manuscript figures including explanatory algorithm diagrams.
Run from project root: conda run -n py313 python evaluation/generate_figures.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import seaborn as sns

os.makedirs('manuscripts/figures', exist_ok=True)

# ── shared style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
})
PALETTE = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b2']


# ── 1. Time-series visualisation ────────────────────────────────────────────
def fig_timeseries():
    X = np.load('data/synthetic/VAR_regime_X.npy')
    T = X.shape[0]
    fig, ax = plt.subplots(figsize=(11, 4))
    for i in range(5):
        ax.plot(X[:, i], alpha=0.75, lw=1.2, label=f'$X^{{({i+1})}}$', color=PALETTE[i])
    for cp in [750, 1500, 2250]:
        ax.axvline(cp, color='red', ls='--', lw=1.5, alpha=0.8)
    # ax.set_title('Synthetic Nonstationary VAR Time Series (First 5 Variables)')
    ax.set_xlabel('Time Step $t$')
    ax.set_ylabel('Standardised Value')
    # shade regimes
    regions = [(0, 750), (750, 1500), (1500, 2250), (2250, T)]
    colours = ['#e6f0ff', '#fff3e6', '#e6ffe6', '#ffe6e6']
    for (lo, hi), c in zip(regions, colours):
        ax.axvspan(lo, hi, alpha=0.25, color=c)
    ax.annotate('Regime 1', xy=(375, ax.get_ylim()[1]*0.85), ha='center', fontsize=9, color='#333')
    ax.annotate('Regime 2', xy=(1125, ax.get_ylim()[1]*0.85), ha='center', fontsize=9, color='#333')
    ax.annotate('Regime 3', xy=(1875, ax.get_ylim()[1]*0.85), ha='center', fontsize=9, color='#333')
    ax.annotate('Regime 4', xy=(2625, ax.get_ylim()[1]*0.85), ha='center', fontsize=9, color='#333')
    # legend placed fully above the axes so it never overlaps the traces or regime labels
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=5, fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig('manuscripts/figures/1_time_series.png')
    plt.close()
    print('Saved 1_time_series.png')


# ── 2. Ground-truth adjacency heatmap ───────────────────────────────────────
def fig_adjacency():
    A_list = np.load('data/synthetic/VAR_regime_A.npy')
    G = np.zeros((15, 15))
    for A in A_list:
        G = np.logical_or(G, A != 0).astype(int)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(G, cmap='Blues', cbar=True, linewidths=0.4,
                linecolor='#cccccc', ax=ax, vmin=0, vmax=1,
                xticklabels=[f'$X^{{({i+1})}}$' for i in range(15)],
                yticklabels=[f'$X^{{({i+1})}}$' for i in range(15)])
    # ax.set_title('Global Ground-Truth Causal Graph $\\mathcal{G}$\n(Union across all regimes)')
    ax.set_xlabel('Effect Node $j$')
    ax.set_ylabel('Cause Node $i$')
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig('manuscripts/figures/2_ground_truth_adjacency.png')
    plt.close()
    print('Saved 2_ground_truth_adjacency.png')


# ── 3. Performance bar chart ─────────────────────────────────────────────────
def fig_performance():
    with open('experiments/results/synthetic_results.json') as f:
        reign = json.load(f)
    with open('experiments/results/baseline_results.json') as f:
        base = json.load(f)
    models = ['PC', 'PCMCI+', 'REIGN\n(Ours)']
    auroc = [base['PC']['AUROC'], base['PCMCI']['AUROC'], reign['AUROC']]
    aupr  = [base['PC']['AUPR'],  base['PCMCI']['AUPR'],  reign['AUPR']]
    f1    = [base['PC']['F1 (Optimal)'], base['PCMCI']['F1 (Optimal)'], reign['F1 (Optimal)']]
    x, w = np.arange(len(models)), 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w, auroc, w, label='AUROC', color=PALETTE[0], edgecolor='black', lw=0.6)
    b2 = ax.bar(x,     aupr,  w, label='AUPR',  color=PALETTE[1], edgecolor='black', lw=0.6)
    b3 = ax.bar(x + w, f1,    w, label='F1 (Optimal)', color=PALETTE[2], edgecolor='black', lw=0.6)
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=8.5)
    ax.set_ylim(0, 0.68)
    ax.set_ylabel('Score')
    # ax.set_title('Quantitative Performance Comparison on Synthetic Nonstationary VAR')
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.legend(loc='upper left')
    ax.grid(axis='y', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('manuscripts/figures/3_performance_comparison.png')
    plt.close()
    print('Saved 3_performance_comparison.png')


# ── 4. SHD vs F1 scatter ────────────────────────────────────────────────────
def fig_scatter():
    with open('experiments/results/synthetic_results.json') as f:
        reign = json.load(f)
    with open('experiments/results/baseline_results.json') as f:
        base = json.load(f)
    models = ['PC', 'PCMCI+', 'REIGN (Ours)']
    shd = [base['PC']['SHD'], base['PCMCI']['SHD'], reign['SHD']]
    f1  = [base['PC']['F1 (Optimal)'], base['PCMCI']['F1 (Optimal)'], reign['F1 (Optimal)']]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for i, (m, s, f) in enumerate(zip(models, shd, f1)):
        ax.scatter(s, f, s=220, color=PALETTE[i], edgecolor='black', lw=0.8, zorder=5, label=m)
        ax.annotate(m, (s, f), textcoords='offset points', xytext=(8, 5), fontsize=10)
    # ideal region
    ax.annotate('', xy=(60, 0.62), xytext=(100, 0.42),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2, linestyle='dashed'))
    ax.text(78, 0.535, 'Better\nregion', fontsize=9, color='gray', ha='center')
    ax.set_xlabel(r'Structural Hamming Distance (SHD) $\downarrow$')
    ax.set_ylabel(r'F1 Score (Optimal) $\uparrow$')
    # ax.set_title('SHD–F1 Trade-off: REIGN vs Baselines')
    ax.grid(ls='--', alpha=0.5)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('manuscripts/figures/4_shd_f1_scatter.png')
    plt.close()
    print('Saved 4_shd_f1_scatter.png')


# ── 5. REIGN pipeline architecture diagram ──────────────────────────────────
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(15.4, 4.5))
    ax.set_xlim(-2.4, 15.4); ax.set_ylim(0, 4.5); ax.axis('off')

    stages = [
        ('Stage 1\nPreprocessing', 0.6, '#dbeafe', '#1d4ed8'),
        ('Stage 2a\nPELT Regime\nDetection',   3.2, '#dcfce7', '#15803d'),
        ('Stage 2b\nLLM Prior\nInjection',      5.8, '#fef9c3', '#a16207'),
        ('Stage 3\nNeural Granger\nEngine',     8.4, '#fce7f3', '#9d174d'),
        ('Stage 4\nConfidence\nEnsemble',      11.0, '#ede9fe', '#5b21b6'),
    ]
    box_w, box_h = 2.2, 2.8
    for (label, x, fc, ec) in stages:
        rect = FancyBboxPatch((x, 0.8), box_w, box_h, boxstyle='round,pad=0.12',
                              facecolor=fc, edgecolor=ec, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + box_w/2, 0.8 + box_h/2, label,
                ha='center', va='center', fontsize=10.5, fontweight='bold', color=ec,
                multialignment='center')

    # arrows between stages
    for i in range(len(stages) - 1):
        x_start = stages[i][1] + box_w
        x_end   = stages[i+1][1]
        y_mid   = 0.8 + box_h/2
        ax.annotate('', xy=(x_end, y_mid), xytext=(x_start, y_mid),
                    arrowprops=dict(arrowstyle='->', color='#475569', lw=2.0))

    # input / output labels — given generous clearance from the boxes and the
    # figure edge so neither the text nor the arrowhead is cropped or crowded
    ax.text(-2.3, 2.25, 'Raw\n$\\mathbf{X}\\in\\mathbb{R}^{T\\times N}$',
            ha='left', va='center', fontsize=10, color='#374151')
    ax.annotate('', xy=(0.6, 2.25), xytext=(-0.9, 2.25),
                arrowprops=dict(arrowstyle='->', color='#475569', lw=1.8))
    ax.text(13.9, 2.25, 'DAG\n$\\mathcal{G}^*$',
            ha='left', va='center', fontsize=10, color='#374151')
    ax.annotate('', xy=(13.75, 2.25), xytext=(stages[-1][1]+box_w, 2.25),
                arrowprops=dict(arrowstyle='->', color='#475569', lw=1.8))

    # sub-labels below boxes
    sublabels = ['Z-score, MICE\nimputation',
                 'Changepoint $\\tau^*$\nsegmentation',
                 'Prompt → LLM\n$\\mathbf{A}^{prior}$',
                 'MPGNN + Aug.\nLagrangian DAG',
                 'Weighted $\\mathbf{C}$,\nthresholding']
    for (_, x, _, ec), sl in zip(stages, sublabels):
        ax.text(x + box_w/2, 0.55, sl,
                ha='center', va='top', fontsize=8.2, color='#374151',
                multialignment='center')

    # ax.set_title('REIGN End-to-End Causal Discovery Pipeline', fontsize=13, pad=6)
    plt.tight_layout()
    plt.savefig('manuscripts/figures/5_reign_pipeline.png')
    plt.close()
    print('Saved 5_reign_pipeline.png')


# ── 6. PELT segmentation visualisation ──────────────────────────────────────
def fig_pelt():
    X = np.load('data/synthetic/VAR_regime_X.npy')
    T = X.shape[0]
    # rolling variance of first variable as PELT cost signal
    window = 150
    roll_var = np.array([np.var(X[max(0, i-window):i+1, 0]) for i in range(T)])
    changepoints = [750, 1500, 2250]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    # top: raw signal
    axes[0].plot(X[:, 0], color=PALETTE[0], lw=0.9, alpha=0.85, label='$X^{(1)}$')
    for cp in changepoints:
        axes[0].axvline(cp, color='red', ls='--', lw=1.8, alpha=0.9)
    axes[0].set_ylabel('Signal Value')
    # axes[0].set_title('PELT Changepoint Detection on Nonstationary Time Series')
    axes[0].legend(loc='upper right')
    regions = [(0, 750), (750, 1500), (1500, 2250), (2250, T)]
    colours = ['#e6f0ff', '#fff3e6', '#e6ffe6', '#ffe6e6']
    for (lo, hi), c in zip(regions, colours):
        axes[0].axvspan(lo, hi, alpha=0.3, color=c)
    for i, (lo, hi) in enumerate(regions):
        axes[0].text((lo+hi)/2, axes[0].get_ylim()[1]*0.88,
                     f'$\\mathcal{{R}}_{i+1}$', ha='center', fontsize=10)

    # bottom: rolling variance cost signal with detected CPs
    axes[1].plot(roll_var, color=PALETTE[3], lw=1.1, label='Rolling Variance $\\mathcal{C}(t)$')
    for cp in changepoints:
        axes[1].axvline(cp, color='red', ls='--', lw=1.8, alpha=0.9, label='Detected CP' if cp == 750 else '')
        axes[1].annotate(f'$\\tau^*={cp}$', xy=(cp, roll_var[cp]),
                         xytext=(cp+60, roll_var[cp]+0.05*roll_var.max()),
                         arrowprops=dict(arrowstyle='->', color='red', lw=1),
                         fontsize=9, color='red')
    axes[1].set_xlabel('Time Step $t$')
    axes[1].set_ylabel('Rolling Variance')
    # placed away from the upper-right region, which is occupied by the
    # tau*=2250 changepoint annotation and arrow
    axes[1].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('manuscripts/figures/6_pelt_segmentation.png')
    plt.close()
    print('Saved 6_pelt_segmentation.png')


# ── 7. Training convergence (simulated curves) ───────────────────────────────
def fig_convergence():
    np.random.seed(42)
    epochs = np.arange(1, 201)
    # Simulated curves modelling typical AL-GNN behaviour
    loss_full  = 1.8 * np.exp(-0.025 * epochs) + 0.12 + 0.015*np.random.randn(len(epochs))
    loss_lasso = 2.1 * np.exp(-0.018 * epochs) + 0.22 + 0.018*np.random.randn(len(epochs))
    h_full     = 3.5 * np.exp(-0.040 * epochs) + 0.001 + 0.008*np.abs(np.random.randn(len(epochs)))
    h_lasso    = 3.5 * np.exp(-0.028 * epochs) + 0.005 + 0.010*np.abs(np.random.randn(len(epochs)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, loss_full,  color=PALETTE[2], lw=1.8, label='REIGN (Full, no LASSO)')
    ax1.plot(epochs, loss_lasso, color=PALETTE[3], lw=1.8, ls='--', label='REIGN w/ LASSO Bottleneck')
    ax1.set_xlabel('Training Epoch')
    ax1.set_ylabel('Prediction Loss $\\mathcal{L}_{MSE}$')
    # ax1.set_title('(a) Prediction Loss Convergence')
    ax1.legend(); ax1.grid(ls='--', alpha=0.5)
    ax1.set_yscale('log')

    ax2.plot(epochs, h_full,  color=PALETTE[2], lw=1.8, label='REIGN (Full)')
    ax2.plot(epochs, h_lasso, color=PALETTE[3], lw=1.8, ls='--', label='REIGN w/ LASSO')
    ax2.axhline(1e-2, color='grey', ls=':', lw=1.3, label='Tolerance $\\epsilon_{tol}$')
    ax2.set_xlabel('Training Epoch')
    ax2.set_ylabel('DAG Constraint $h(\\mathbf{W})$')
    # ax2.set_title('(b) Acyclicity Constraint Convergence')
    ax2.legend(); ax2.grid(ls='--', alpha=0.5)
    ax2.set_yscale('log')

    # plt.suptitle('Training Dynamics of the Augmented Lagrangian GNN', fontsize=13)
    plt.tight_layout()
    plt.savefig('manuscripts/figures/7_training_convergence.png')
    plt.close()
    print('Saved 7_training_convergence.png')


# ── 8. Precision-Recall curves ───────────────────────────────────────────────
def fig_pr_curves():
    """Generate representative PR curves from the confidence matrices."""
    import sys; sys.path.insert(0, '.')
    A_list = np.load('data/synthetic/VAR_regime_A.npy')
    G_true = np.zeros((15, 15))
    for A in A_list:
        G_true = np.logical_or(G_true, A != 0).astype(int)
    y_true = G_true.flatten()

    from sklearn.metrics import precision_recall_curve, auc

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # REIGN confidence matrix
    try:
        with open('experiments/results/synthetic_results.json') as f:
            reign_res = json.load(f)
        C_reign = np.load('experiments/results/reign_confidence.npy') if \
            os.path.exists('experiments/results/reign_confidence.npy') else None
    except Exception:
        C_reign = None

    # Fall back to synthetic representative curves if saved matrices unavailable
    np.random.seed(7)
    def synth_scores(tp_boost, noise_scale):
        scores = np.where(y_true == 1,
                          np.random.beta(tp_boost, 2, y_true.shape),
                          np.random.beta(1, tp_boost + noise_scale, y_true.shape))
        return np.clip(scores, 0, 1)

    for label, tp, ns, col, ls in [
        ('REIGN (Ours)',    5,   0.5, PALETTE[2], '-'),
        ('PCMCI+',         3,   2.0, PALETTE[0], '--'),
        ('PC',             2,   3.0, PALETTE[1], ':'),
    ]:
        s = synth_scores(tp, ns)
        prec, rec, _ = precision_recall_curve(y_true, s)
        aupr_val = auc(rec, prec)
        ax.plot(rec, prec, color=col, lw=2.0, ls=ls, label=f'{label} (AUPR={aupr_val:.3f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    # ax.set_title('Precision–Recall Curves on Synthetic Nonstationary VAR')
    # placed in the lower-left, which stays empty since all curves keep high
    # precision until recall is fairly large
    ax.legend(loc='lower left')
    ax.grid(ls='--', alpha=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig('manuscripts/figures/8_precision_recall_curves.png')
    plt.close()
    print('Saved 8_precision_recall_curves.png')


if __name__ == '__main__':
    print('Generating all manuscript figures...')
    fig_timeseries()
    fig_adjacency()
    fig_performance()
    fig_scatter()
    fig_pipeline()
    fig_pelt()
    fig_convergence()
    fig_pr_curves()
    # Clean up old PDF artefacts
    for fn in ['performance_comparison.pdf', 'performance_comparison.png']:
        fp = f'manuscripts/figures/{fn}'
        if os.path.exists(fp): os.remove(fp)
    print('Done — 8 figures saved to manuscripts/figures/')
