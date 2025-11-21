from dagster import asset
import pandas as pd

@asset
def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the raw spotify dataset."""
    """
    Performs basic cleaning operations on the raw dataset.
    - Drop NA values
    """
    df = raw_df.copy()
    df.dropna(inplace=True)

    return df