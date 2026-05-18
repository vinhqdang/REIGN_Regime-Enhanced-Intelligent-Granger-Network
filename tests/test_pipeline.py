import numpy as np
import pytest
from src.preprocessing import preprocess
from src.regime_detection import detect_regimes_PELT
from src.reign import REIGN

def test_preprocessing():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    timestamps = np.arange(100)
    
    X_clean = preprocess(X, timestamps)
    assert X_clean.shape == (100, 5)
    assert not np.isnan(X_clean).any()

def test_regime_detection():
    np.random.seed(42)
    # Generate 2 distinct regimes
    X1 = np.random.randn(50, 2)
    X2 = np.random.randn(50, 2) + 5
    X = np.concatenate([X1, X2])
    
    regimes = detect_regimes_PELT(X, min_regime_length=20)
    assert len(regimes) > 0
    assert regimes[-1][1] == 100

def test_reign_pipeline():
    np.random.seed(42)
    X = np.random.randn(100, 3)
    timestamps = np.arange(100)
    variable_names = ["var_0", "var_1", "var_2"]
    variable_descriptions = ["Desc 0", "Desc 1", "Desc 2"]
    domain_desc = "Test domain"
    
    G_star, confidence, stability = REIGN(
        X, timestamps, variable_names, variable_descriptions, domain_desc,
        use_mock_llm=True
    )
    
    assert G_star.shape == (3, 3)
    assert confidence.shape == (3, 3)
