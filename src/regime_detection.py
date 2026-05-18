import numpy as np
import ruptures as rpt

def compute_rolling_features(X: np.ndarray, window: int = 10) -> np.ndarray:
    """Compute rolling mean and variance to augment features for PELT."""
    T, N = X.shape
    # Pad at the beginning to maintain length
    padded_X = np.pad(X, ((window-1, 0), (0, 0)), mode='edge')
    
    means = np.zeros_like(X)
    variances = np.zeros_like(X)
    
    for t in range(T):
        window_slice = padded_X[t:t+window]
        means[t] = np.mean(window_slice, axis=0)
        variances[t] = np.var(window_slice, axis=0)
        
    # Concatenate original X, means, and variances
    return np.concatenate([X, means, variances], axis=1)

def merge_short_regimes(changepoints: list, T: int, min_regime_length: int = 30) -> list:
    """Merge regimes shorter than min_regime_length."""
    # ruptures returns changepoints ending with T (e.g., [100, 200, T])
    if len(changepoints) == 0 or (len(changepoints) == 1 and changepoints[0] == T):
        return [T]
        
    filtered_cps = []
    prev_cp = 0
    for cp in changepoints:
        length = cp - prev_cp
        if length >= min_regime_length or cp == T:
            filtered_cps.append(cp)
            prev_cp = cp
        # Else: we drop this cp, effectively merging it with the next
    
    if filtered_cps[-1] != T:
        filtered_cps.append(T)
        
    # Second pass: check if the last regime is too short, if so merge with previous
    if len(filtered_cps) > 1 and (filtered_cps[-1] - filtered_cps[-2]) < min_regime_length:
        filtered_cps.pop(-2)
        
    return filtered_cps

def detect_regimes_PELT(X: np.ndarray, min_regime_length: int = 30, penalty: float = None) -> list:
    """
    Stage 2a — Regime Segmentation (PELT)
    1. Compute per-variable rolling mean + variance features
    2. Run PELT with RBF cost function, penalty β = log(T') × N
    3. Merge regimes shorter than min_regime_length
    
    Returns:
        List of tuples: [(start_1, end_1), (start_2, end_2), ...]
    """
    T, N = X.shape
    if T < min_regime_length:
        return [(0, T)]
        
    features = compute_rolling_features(X)
    
    if penalty is None:
        penalty = np.log(T) * N
        
    # Run Pelt
    algo = rpt.Pelt(model="rbf", min_size=min_regime_length).fit(features)
    changepoints = algo.predict(pen=penalty)
    
    merged_cps = merge_short_regimes(changepoints, T, min_regime_length)
    
    # Convert to list of (start, end) tuples
    regimes = []
    start = 0
    for cp in merged_cps:
        regimes.append((start, cp))
        start = cp
        
    return regimes
