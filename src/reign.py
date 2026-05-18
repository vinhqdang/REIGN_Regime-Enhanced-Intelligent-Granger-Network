from .preprocessing import preprocess
from .regime_detection import detect_regimes_PELT
from .llm_prior import generate_llm_prior
from .cuts_plus import coarse_discovery_VAR, train_mpgnn_granger, threshold_graph
from .ensemble import confidence_weighted_ensemble

def REIGN(X, timestamps, variable_names, variable_descriptions, domain_desc,
          lambda_prior=0.1, alpha_dag=1.0, gamma_sparse=0.01, use_mock_llm=True):
    """
    Main REIGN pipeline: Regime-Enhanced Intelligent Granger Network.
    """
    # Stage 1: Preprocessing
    print("Stage 1: Preprocessing data...")
    X_clean = preprocess(X, timestamps, variable_names)

    # Stage 2a: Regime segmentation
    print("Stage 2a: Detecting regimes...")
    regimes = detect_regimes_PELT(X_clean)
    print(f"Detected {len(regimes)} regimes.")

    # Stage 2b: LLM prior
    print("Stage 2b: Generating LLM prior...")
    A_llm, conf_matrix = generate_llm_prior(
        variable_names, variable_descriptions, domain_desc, use_mock=use_mock_llm
    )

    # Stage 3: Per-regime CUTS+ Granger
    print("Stage 3: Training per-regime CUTS+ Granger models...")
    local_graphs = []
    for k, (start, end) in enumerate(regimes):
        print(f"  Regime {k+1}/{len(regimes)} (steps {start} to {end})")
        X_k = X_clean[start:end]
        
        # If regime is too short or coarse discovery fails gracefully
        try:
            E_coarse = coarse_discovery_VAR(X_k)
            W_k = train_mpgnn_granger(X_k, E_coarse, A_llm, confidence=conf_matrix,
                                       lambda_prior=lambda_prior, alpha_dag=alpha_dag, gamma_sparse=gamma_sparse)
            G_k = threshold_graph(W_k, tau=0.1)
            local_graphs.append((G_k, W_k, end - start))
        except Exception as e:
            print(f"    Error in regime {k+1}: {e}. Skipping.")

    # Stage 4: Ensemble
    print("Stage 4: Confidence-weighted ensemble...")
    G_star, confidence, stability_labels = confidence_weighted_ensemble(local_graphs)

    print("REIGN execution completed.")
    return G_star, confidence, stability_labels
