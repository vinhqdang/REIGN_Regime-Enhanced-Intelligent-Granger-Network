import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

def preprocess(X: np.ndarray, timestamps: np.ndarray, variable_names: list = None) -> np.ndarray:
    """
    Stage 1 — Preprocessing & Imputation
    1. Z-score normalize each variable independently
    2. Detect and flag outliers (IQR ×3 threshold)
    3. Resample to uniform grid via linear interpolation
    4. Impute remaining NaN via forward-fill + MICE
    """
    if variable_names is None:
        variable_names = [f"var_{i}" for i in range(X.shape[1])]
        
    # Convert timestamps to datetime if they are not already
    if not isinstance(timestamps[0], (pd.Timestamp, np.datetime64)):
        # Assume numeric timestamps (e.g., seconds or milliseconds)
        index = pd.to_datetime(timestamps, unit='s') # Default to seconds
    else:
        index = pd.to_datetime(timestamps)
        
    df = pd.DataFrame(X, index=index, columns=variable_names)
    
    # 1. Z-score normalize
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
    
    # 2. Detect and flag outliers (IQR x3)
    Q1 = df_scaled.quantile(0.25)
    Q3 = df_scaled.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    mask = (df_scaled < lower_bound) | (df_scaled > upper_bound)
    df_scaled[mask] = np.nan
    
    # 3. Resample to uniform grid
    if len(df) > 1:
        gaps = pd.Series(df.index).diff().dropna()
        median_gap = gaps.median()
        
        if pd.isna(median_gap) or median_gap == pd.Timedelta(seconds=0):
            median_gap = pd.Timedelta(seconds=1)
            
        df_resampled = df_scaled.resample(median_gap).mean()
        df_resampled = df_resampled.interpolate(method='linear')
    else:
        df_resampled = df_scaled

    # 4. Impute remaining NaN via forward-fill + MICE
    df_resampled = df_resampled.ffill(limit=3)
    
    if df_resampled.isna().sum().sum() > 0:
        imputer = IterativeImputer(random_state=42, max_iter=10)
        imputed_data = imputer.fit_transform(df_resampled)
        df_final = pd.DataFrame(imputed_data, index=df_resampled.index, columns=df_resampled.columns)
    else:
        df_final = df_resampled
        
    return df_final.values
