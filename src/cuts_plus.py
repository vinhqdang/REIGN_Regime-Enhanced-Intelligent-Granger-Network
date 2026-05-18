import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Lasso

class MPGNN(nn.Module):
    def __init__(self, num_vars, lag=5, hidden_dim=64):
        super().__init__()
        self.num_vars = num_vars
        self.lag = lag
        
        self.W = nn.Parameter(torch.zeros(num_vars, num_vars))
        nn.init.uniform_(self.W, -0.1, 0.1)
        
        self.node_mlp = nn.Sequential(
            nn.Linear(lag, hidden_dim),
            nn.ReLU()
        )
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.out_mlp = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        h = self.node_mlp(x) 
        
        out_preds = []
        for i in range(self.num_vars):
            h_i = h[:, i, :]
            msg_sum = torch.zeros_like(h_i)
            
            for j in range(self.num_vars):
                if i == j: continue
                h_j = h[:, j, :]
                
                cat_hj = torch.cat([h_i, h_j], dim=-1)
                m_ij = self.edge_mlp(cat_hj)
                
                msg_sum += m_ij * self.W[i, j]
                
            agg_h = h_i + msg_sum
            out_preds.append(self.out_mlp(agg_h))
            
        return torch.cat(out_preds, dim=-1)

def coarse_discovery_VAR(X: np.ndarray, lag: int = 5, threshold: float = 1e-4) -> set:
    T, N = X.shape
    return set((i, j) for i in range(N) for j in range(N))

def notears_h(W):
    W_sq = W * W
    E = torch.matrix_exp(W_sq)
    return torch.trace(E) - W.shape[0]

def train_mpgnn_granger(X: np.ndarray, E_coarse: set, A_LLM: np.ndarray, confidence: np.ndarray = None,
                       lambda_prior: float = 0.1, alpha_dag: float = 1.0, gamma_sparse: float = 0.01,
                       lag: int = 5, epochs: int = 200) -> np.ndarray:
    torch.manual_seed(42)
    np.random.seed(42)
    T, N = X.shape
    if T <= lag:
        return np.zeros((N, N))
        
    if confidence is None:
        confidence = np.ones((N, N))
        
    model = MPGNN(N, lag)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    X_seq = []
    y_seq = []
    for t in range(lag, T):
        X_seq.append(X[t-lag:t].T) 
        y_seq.append(X[t])         
        
    X_t = torch.FloatTensor(np.array(X_seq))
    y_t = torch.FloatTensor(np.array(y_seq))
    A_LLM_t = torch.FloatTensor(A_LLM)
    conf_t = torch.FloatTensor(confidence)
    
    rho = 1.0
    alpha = 0.0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        preds = model(X_t)
        loss_pred = F.mse_loss(preds, y_t)
        
        W = model.W
        
        mask = torch.ones_like(W)
        for i in range(N):
            for j in range(N):
                if (i, j) not in E_coarse: 
                    mask[i, j] = 0
                    
        W_masked = W * mask
        
        loss_prior = torch.sum(conf_t * (W_masked - A_LLM_t)**2)
        loss_sparse = torch.norm(W_masked, p=1)
        h_val = notears_h(W_masked)
        
        loss = loss_pred + lambda_prior * loss_prior + gamma_sparse * loss_sparse
        loss += 0.5 * rho * h_val * h_val + alpha * h_val
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            if epoch % 10 == 0:
                h_new = notears_h(model.W * mask).item()
                if h_new > 0.25 * h_val.item():
                    rho *= 10
                alpha += rho * h_new
                
    final_W = (model.W * mask).detach().numpy()
    return final_W

def threshold_graph(W: np.ndarray, tau: float = 0.1) -> np.ndarray:
    return (np.abs(W) > tau).astype(int)
