import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def setup_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.5)

def generate_performance_bar_chart():
    results_dir = "experiments/results"
    with open(os.path.join(results_dir, "synthetic_results.json"), "r") as f:
        reign_res = json.load(f)
    with open(os.path.join(results_dir, "baseline_results.json"), "r") as f:
        baselines_res = json.load(f)
        
    models = ['PC', 'PCMCI+', 'REIGN (Ours)']
    auroc = [baselines_res['PC']['AUROC'], baselines_res['PCMCI']['AUROC'], reign_res['AUROC']]
    aupr = [baselines_res['PC']['AUPR'], baselines_res['PCMCI']['AUPR'], reign_res['AUPR']]
    f1 = [baselines_res['PC']['F1 (Optimal)'], baselines_res['PCMCI']['F1 (Optimal)'], reign_res['F1 (Optimal)']]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, auroc, width, label='AUROC', color='#4c72b0', edgecolor='black')
    rects2 = ax.bar(x, aupr, width, label='AUPR', color='#dd8452', edgecolor='black')
    rects3 = ax.bar(x + width, f1, width, label='F1 (Optimal)', color='#55a868', edgecolor='black')
    
    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison on Synthetic Nonstationary VAR')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig('manuscripts/figures/3_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_time_series_plot():
    X = np.load('data/synthetic/VAR_regime_X.npy')
    # Plot first 5 variables
    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(min(5, X.shape[1])):
        ax.plot(X[:, i], alpha=0.7, label=f'v{i}')
        
    ax.axvline(x=1500, color='r', linestyle='--', linewidth=2, label='True Regime Shift')
    ax.set_title('Nonstationary Multivariate Time Series (First 5 Variables)')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('manuscripts/figures/1_time_series.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_adjacency_heatmap():
    A_list = np.load('data/synthetic/VAR_regime_A.npy')
    # A_list shape is (n_regimes, N, N)
    G_true = np.zeros((15, 15))
    for A in A_list: 
        G_true = np.logical_or(G_true, (A != 0)).astype(int)
        
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(G_true, cmap='Blues', cbar=False, linewidths=0.5, linecolor='gray', ax=ax)
    ax.set_title('Ground Truth Global Causal Graph (Adjacency Matrix)')
    ax.set_xlabel('Effect Node')
    ax.set_ylabel('Cause Node')
    plt.tight_layout()
    plt.savefig('manuscripts/figures/2_ground_truth_adjacency.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_scatter_plot():
    results_dir = "experiments/results"
    with open(os.path.join(results_dir, "synthetic_results.json"), "r") as f:
        reign_res = json.load(f)
    with open(os.path.join(results_dir, "baseline_results.json"), "r") as f:
        baselines_res = json.load(f)
        
    models = ['PC', 'PCMCI+', 'REIGN (Ours)']
    shd = [baselines_res['PC']['SHD'], baselines_res['PCMCI']['SHD'], reign_res['SHD']]
    f1 = [baselines_res['PC']['F1 (Optimal)'], baselines_res['PCMCI']['F1 (Optimal)'], reign_res['F1 (Optimal)']]
    colors = ['#4c72b0', '#dd8452', '#55a868']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, model in enumerate(models):
        ax.scatter(shd[i], f1[i], s=200, label=model, color=colors[i], edgecolor='black', zorder=5)
        
    ax.set_xlabel(r'Structural Hamming Distance (SHD) $\downarrow$')
    ax.set_ylabel(r'F1 Score $\uparrow$')
    ax.set_title('Trade-off between SHD and F1 Score')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig('manuscripts/figures/4_shd_f1_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    os.makedirs('manuscripts/figures', exist_ok=True)
    setup_style()
    generate_time_series_plot()
    generate_adjacency_heatmap()
    generate_performance_bar_chart()
    generate_scatter_plot()
    
    # Clean up old pdfs if they exist
    if os.path.exists('manuscripts/figures/performance_comparison.pdf'):
        os.remove('manuscripts/figures/performance_comparison.pdf')
    if os.path.exists('manuscripts/figures/performance_comparison.png'):
        os.remove('manuscripts/figures/performance_comparison.png')
        
    print("Generated 4 high-quality PNG figures for the manuscript.")
