"""Feature engineering assets."""

import pandas as pd
from dagster import asset, get_dagster_logger
from ml_pipeline.utils.feature_engineering import (
    engineer_features,
    prepare_features_and_target,
)


@asset
def engineered_train_features(train_data: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for training data.
    
    Args:
        train_data: Training dataset.
        
    Returns:
        DataFrame with engineered features.
    """
    logger = get_dagster_logger()
    logger.info("Engineering features for training data...")
    df = engineer_features(train_data)
    return df


@asset
def engineered_test_features(test_data: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for test data.
    
    Args:
        test_data: Test dataset.
        
    Returns:
        DataFrame with engineered features.
    """
    logger = get_dagster_logger()
    logger.info("Engineering features for test data...")
    df = engineer_features(test_data)
    return df


@asset
def train_features_and_target(engineered_train_features: pd.DataFrame) -> tuple:
    """Prepare training features and target.
    
    Args:
        engineered_train_features: Engineered training features.
        
    Returns:
        Tuple of (features, target).
    """
    X_train, y_train = prepare_features_and_target(
        engineered_train_features,
        target_column="target"  # Adjust column name
    )
    return X_train, y_train


@asset
def test_features_and_target(engineered_test_features: pd.DataFrame) -> tuple:
    """Prepare test features and target.
    
    Args:
        engineered_test_features: Engineered test features.
        
    Returns:
        Tuple of (features, target).
    """
    X_test, y_test = prepare_features_and_target(
        engineered_test_features,
        target_column="target"  # Adjust column name
    )
    return X_test, y_test