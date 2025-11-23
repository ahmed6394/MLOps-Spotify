from dagster import asset
import pandas as pd

@asset
def clean_data(load_data: pd.DataFrame) -> pd.DataFrame:
    """Cleans the raw spotify dataset."""
    """
    Performs basic cleaning operations on the raw dataset.
    - Drop NA values
    """
    df = load_data.copy()
    df.dropna(inplace=True)

    return df