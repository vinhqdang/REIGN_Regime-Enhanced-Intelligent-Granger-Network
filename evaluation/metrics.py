import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def shd(G_pred: np.ndarray, G_true: np.ndarray) -> int:
    """Structural Hamming Distance = FP + FN + reversals"""
    # Simply sum absolute differences for binary DAGs
    diff = np.abs(G_pred - G_true)
    return int(np.sum(diff))

def f1(G_pred: np.ndarray, G_true: np.ndarray) -> float:
    TP = np.sum((G_pred == 1) & (G_true == 1))
    FP = np.sum((G_pred == 1) & (G_true == 0))
    FN = np.sum((G_pred == 0) & (G_true == 1))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def evaluate(C: np.ndarray, G_pred: np.ndarray, G_true: np.ndarray):
    """
    Compute threshold-free AUROC and other structural metrics.
    """
    y_true = G_true.flatten()
    y_score = C.flatten()
    
    # Compute Max-F1 across all thresholds
    max_c = np.max(C) if np.max(C) > 0 else 1.0
    thresholds = np.linspace(0.0, max_c + 0.1, 100)
    best_f1 = 0.0
    best_shd = float('inf')
    
    for tau in thresholds:
        pred_tau = (C > tau).astype(int)
        current_f1 = f1(pred_tau, G_true)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_shd = shd(pred_tau, G_true)
            
    # Fallback to default threshold if Max-F1 is 0
    if best_f1 == 0.0:
        best_shd = shd(G_pred, G_true)
    
    # Check if y_true has more than one class to compute AUROC/AUPR
    if len(np.unique(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_score)
        aupr = average_precision_score(y_true, y_score)
    else:
        auroc = float('nan')
        aupr = float('nan')
        
    return {
        'AUROC': auroc,
        'AUPR': aupr,
        'SHD': best_shd,
        'F1 (Optimal)': best_f1,
    }
