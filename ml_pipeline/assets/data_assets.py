"""Data loading and preprocessing assets."""

from dagster import asset
import pandas as pd
import kagglehub

@asset
def load_data():
    """Load and preprocess the Spotify tracks dataset."""
    data_path = kagglehub.dataset_download("amitanshjoshi/spotify-1million-tracks")
    df = pd.read_csv(f"{data_path}/spotify_data.csv")

    return df