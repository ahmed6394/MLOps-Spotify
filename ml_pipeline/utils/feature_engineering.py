"""Feature engineering utilities."""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering transformations.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        DataFrame with engineered features.
    """
    # Copy your feature engineering logic here
    # Example:
    df['feature_1'] = df['col_a'] * df['col_b']
    return df


def prepare_features_and_target(
    df: pd.DataFrame,
    target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features and target variable.
    
    Args:
        df: Input DataFrame.
        target_column: Name of target column.
        
    Returns:
        Tuple of (features, target).
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def scale_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Standardize features.
    
    Args:
        X: Feature DataFrame.
        
    Returns:
        Tuple of (scaled features, scaler object).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns), scaler