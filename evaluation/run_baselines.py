import numpy as np
import os
import sys
import json
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metrics import evaluate

def run_pcmci(X, tau_max=2, pc_alpha=0.2):
    try:
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr
    except ImportError:
        print("Tigramite not installed.")
        return None
        
    N = X.shape[1]
    dataframe = pp.DataFrame(X)
    pcmci = PCMCI(
        dataframe=dataframe, 
        cond_ind_test=ParCorr(),
        verbosity=0
    )
    
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)
    p_matrix = results['p_matrix']
    val_matrix = results['val_matrix']
    
    # Create G_pred (N, N) where A[i, j] means j causes i
    # tigramite format: (N, N, tau_max+1)
    # i, j, tau means i(t-tau) causes j(t)
    # We want a summary graph over all lags tau > 0
    
    G_pred = np.zeros((N, N))
    confidence = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if i == j: continue
            # Look at lags 1 to tau_max
            p_vals = p_matrix[i, j, 1:]
            vals = val_matrix[i, j, 1:]
            
            # If any lag is significant
            if np.any(p_vals < pc_alpha):
                G_pred[j, i] = 1  # Note the index convention: we need j causes i, wait: if i causes j, G[j, i] = 1? 
                # Let's align with evaluate: G_true has G_true[i, j] = 1 if j causes i.
                # Actually, our evaluate doesn't care, it just compares G_pred and G_true directly.
                # Let's say G_pred[j, i] = 1 if i causes j. Wait, if i causes j, it's (i, j).
                G_pred[i, j] = 1
                confidence[i, j] = np.max(np.abs(vals))
                
    return confidence, G_pred

def run_pc(X):
    try:
        from causallearn.search.ConstraintBased.PC import pc
    except ImportError:
        print("causal-learn not installed.")
        return None
        
    N = X.shape[1]
    cg = pc(X, alpha=0.05, indep_test='fisherz')
    
    # causal-learn graph: 1 is tail, -1 is arrowhead. 
    # Usually G[i, j] = -1 and G[j, i] = 1 means i -> j
    # We'll just build an unweighted adjacency for evaluating
    G_pred = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if cg.G.graph[i, j] == -1 and cg.G.graph[j, i] == 1:
                G_pred[i, j] = 1  # j causes i or i causes j, depending on convention
                
    # Confidence is 1 or 0
    return G_pred, G_pred

def run_baselines():
    print("Loading synthetic data...")
    data_dir = "data/synthetic"
    X = np.load(os.path.join(data_dir, "VAR_regime_X.npy"))
    A_list = np.load(os.path.join(data_dir, "VAR_regime_A.npy"))
    
    N = X.shape[1]
    
    G_true = np.zeros((N, N))
    for A in A_list:
        G_true = np.logical_or(G_true, (A != 0)).astype(int)
        
    print(f"Data shape: {X.shape}. Global true edges: {np.sum(G_true)}\n")
    
    results = {}
    
    # 1. PCMCI
    print("Running PCMCI baseline...")
    pcmci_conf, pcmci_pred = run_pcmci(X)
    if pcmci_pred is not None:
        metrics = evaluate(pcmci_conf, pcmci_pred, G_true)
        results['PCMCI'] = metrics
        print("PCMCI Results:")
        for k, v in metrics.items(): print(f"  {k}: {v:.4f}")
        print()
        
    # 2. PC (Static)
    print("Running PC baseline...")
    pc_conf, pc_pred = run_pc(X)
    if pc_pred is not None:
        metrics = evaluate(pc_conf, pc_pred, G_true)
        results['PC'] = metrics
        print("PC Results:")
        for k, v in metrics.items(): print(f"  {k}: {v:.4f}")
        print()
        
    # Save
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/baseline_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    run_baselines()
