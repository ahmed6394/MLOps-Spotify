"""Data loading and preprocessing assets."""

import pandas as pd
from dagster import asset, get_dagster_logger
from ml_pipeline.utils.preprocessing import (
    load_raw_data,
    clean_data,
    split_data,
)


@asset
def raw_data() -> pd.DataFrame:
    """Load raw data from CSV.
    
    Returns:
        Raw DataFrame.
    """
    logger = get_dagster_logger()
    logger.info("Loading raw data...")
    df = load_raw_data("data/raw/spotify_data.csv")  # Adjust path
    logger.info(f"Loaded {len(df)} records")
    return df


@asset
def cleaned_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess raw data.
    
    Args:
        raw_data: Raw input data.
        
    Returns:
        Cleaned DataFrame.
    """
    logger = get_dagster_logger()
    logger.info("Cleaning data...")
    df = clean_data(raw_data)
    logger.info(f"Cleaned data has {len(df)} records")
    return df


@asset
def train_data(cleaned_data: pd.DataFrame) -> pd.DataFrame:
    """Create training dataset.
    
    Args:
        cleaned_data: Cleaned input data.
        
    Returns:
        Training DataFrame.
    """
    logger = get_dagster_logger()
    train, _ = split_data(cleaned_data, test_size=0.2)
    logger.info(f"Training set has {len(train)} records")
    return train


@asset
def test_data(cleaned_data: pd.DataFrame) -> pd.DataFrame:
    """Create test dataset.
    
    Args:
        cleaned_data: Cleaned input data.
        
    Returns:
        Test DataFrame.
    """
    logger = get_dagster_logger()
    _, test = split_data(cleaned_data, test_size=0.2)
    logger.info(f"Test set has {len(test)} records")
    return test