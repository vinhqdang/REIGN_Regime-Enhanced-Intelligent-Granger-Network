import numpy as np
import networkx as nx
import os

def random_sparse_DAG(N=15, density=0.2, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            if np.random.rand() < density:
                # Assign random weight
                weight = np.random.uniform(0.3, 0.8)
                if np.random.rand() < 0.5:
                    weight *= -1
                A[i, j] = weight
    
    # Shuffle nodes to hide topological order
    permutation = np.random.permutation(N)
    P = np.zeros((N, N))
    P[np.arange(N), permutation] = 1
    A = P.T @ A @ P
        
    return A

def simulate_VAR(A, T=750, noise_std=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    N = A.shape[0]
    X = np.zeros((T, N))
    
    # X[t] = A.T @ X[t-1] + noise
    # We transpose A because A[i, j] means i causes j (in some conventions),
    # but let's assume A[i, j] means j causes i here, so X_t = A @ X_{t-1} + e
    
    for t in range(1, T):
        noise = np.random.normal(0, noise_std, N)
        X[t] = A @ X[t-1] + noise
        
    return X

def generate_var_regime_data(N=15, T_per_regime=750, num_regimes=4, output_dir="data/synthetic"):
    os.makedirs(output_dir, exist_ok=True)
    
    regimes = []
    ground_truths = []
    
    np.random.seed(42)
    
    for k in range(num_regimes):
        A_k = random_sparse_DAG(N=N, density=0.2)
        ground_truths.append(A_k)
        
        X_k = simulate_VAR(A_k, T=T_per_regime)
        regimes.append(X_k)
        
    X_full = np.concatenate(regimes, axis=0)
    
    # Save data
    np.save(os.path.join(output_dir, "VAR_regime_X.npy"), X_full)
    np.save(os.path.join(output_dir, "VAR_regime_A.npy"), np.array(ground_truths))
    
    timestamps = np.arange(len(X_full))
    np.save(os.path.join(output_dir, "VAR_regime_timestamps.npy"), timestamps)
    
    print(f"Generated {num_regimes} regimes with {T_per_regime} steps each. Total T={len(X_full)}, N={N}.")
    print(f"Saved to {output_dir}")

if __name__ == "__main__":
    generate_var_regime_data()
