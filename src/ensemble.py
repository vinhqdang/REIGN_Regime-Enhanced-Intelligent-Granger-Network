import numpy as np
import networkx as nx

def enforce_dag(G_star: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """Enforce acyclicity on G* via greedy cycle-breaking."""
    G_dag = G_star.copy()
    
    while True:
        nx_graph = nx.DiGraph(G_dag)
        try:
            cycle = nx.find_cycle(nx_graph, orientation='original')
        except nx.NetworkXNoCycle:
            break
            
        min_conf = float('inf')
        min_edge = None
        for u, v, _ in cycle:
            if confidence[u, v] < min_conf:
                min_conf = confidence[u, v]
                min_edge = (u, v)
                
        if min_edge:
            G_dag[min_edge[0], min_edge[1]] = 0
            
    return G_dag

def confidence_weighted_ensemble(local_graphs: list, tau_ensemble: float = 0.04) -> tuple:
    """
    Stage 4 — Confidence-Weighted Ensemble
    local_graphs: list of tuples (G_k, W_k, T_k)
    """
    if not local_graphs:
        return np.array([]), np.array([]), {}
        
    N = local_graphs[0][0].shape[0]
    total_T = sum(T_k for _, _, T_k in local_graphs)
    K = len(local_graphs)
    
    C = np.zeros((N, N))
    sum_w = 0.0
    
    for G_k, W_k, T_k in local_graphs:
        uncertainty_k = 1.0 
        w_k = (T_k / total_T) * (1 / uncertainty_k)
        
        C += w_k * np.abs(W_k)
        sum_w += w_k
        
    if sum_w > 0:
        C = C / sum_w
        
    G_star = (C > tau_ensemble).astype(int)
    
    G_star = enforce_dag(G_star, C)
    
    edge_presence_count = np.zeros((N, N))
    for G_k, _, _ in local_graphs:
        edge_presence_count += G_k
        
    presence_ratio = edge_presence_count / K
    stability_labels = {}
    
    for i in range(N):
        for j in range(N):
            if G_star[i, j] == 1:
                ratio = presence_ratio[i, j]
                if ratio > 0.8:
                    stability_labels[(i, j)] = "stable"
                elif ratio >= 0.2:
                    stability_labels[(i, j)] = "transient"
                else:
                    stability_labels[(i, j)] = "spurious"
                    G_star[i, j] = 0
                    
    return G_star, C, stability_labels
