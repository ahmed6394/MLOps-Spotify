"""Data preprocessing utilities."""

import pandas as pd
from pathlib import Path
from typing import Tuple


def load_raw_data(data_path: str) -> pd.DataFrame:
    """Load raw data from CSV file.
    
    Args:
        data_path: Path to the CSV file.
        
    Returns:
        DataFrame containing raw data.
    """
    df = pd.read_csv(data_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess data.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Cleaned DataFrame.
    """
    # Copy your preprocessing logic from your notebook here
    df = df.dropna()
    # Add your specific cleaning steps
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets.
    
    Args:
        df: Input DataFrame.
        test_size: Test set proportion.
        random_state: Random seed.
        
    Returns:
        Tuple of (train_df, test_df).
    """
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )
    return train, test