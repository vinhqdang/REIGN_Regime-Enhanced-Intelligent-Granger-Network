import numpy as np
import os
import sys
import json

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reign import REIGN
from evaluation.metrics import evaluate

def run_synthetic_experiment():
    print("Loading synthetic data...")
    data_dir = "data/synthetic"
    X = np.load(os.path.join(data_dir, "VAR_regime_X.npy"))
    A_list = np.load(os.path.join(data_dir, "VAR_regime_A.npy"))
    timestamps = np.load(os.path.join(data_dir, "VAR_regime_timestamps.npy"))
    
    N = X.shape[1]
    
    # Create variable names and descriptions
    variable_names = [f"var_{i}" for i in range(N)]
    variable_descriptions = [f"Synthetic variable {i}" for i in range(N)]
    domain_desc = "Synthetic VAR simulation"
    
    # True global DAG (union of all regimes)
    G_true = np.zeros((N, N))
    for A in A_list:
        G_true = np.logical_or(G_true, (A != 0)).astype(int)
        
    print(f"Data shape: {X.shape}. Global true edges: {np.sum(G_true)}")
    
    print("\nRunning REIGN pipeline...")
    # Use mock LLM for testing
    G_star, confidence, stability = REIGN(
        X, timestamps, variable_names, variable_descriptions, domain_desc,
        lambda_prior=0.0, alpha_dag=1.0, gamma_sparse=0.0, use_mock_llm=True
    )
    
    print("\nEvaluating results...")
    metrics = evaluate(confidence, G_star, G_true)
    
    print("\nResults:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
        
    # Save results
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/synthetic_results.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Results saved to experiments/results/synthetic_results.json")

if __name__ == "__main__":
    run_synthetic_experiment()
