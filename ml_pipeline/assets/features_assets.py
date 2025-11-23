from dagster import asset
import pandas as pd

POPULARITY_THRESHOLD = 85


@asset(
    description="Performs feature engineering and removes unused columns."
)
def feature_engineered_data(clean_data: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering:
    - artist_song_count
    - yearly popularity threshold → binary verdict
    - duration quantile features
    - track_age
    - genre normalization
    - drop unused columns
    """

    df = clean_data.copy()

    # ---------------------------------------
    # Artist-wise song count
    # ---------------------------------------
    if "artist_name" in df.columns and "track_id" in df.columns:
        df['artist_song_count'] = df.groupby('artist_name')['track_id'].transform('count')


    # ---------------------------------------
    # Track age (if 'year' exists)
    # ---------------------------------------
    if "year" in df.columns:
        df["year"] = df["year"].astype(int)
        df["track_age"] = 2025 - df["year"]


    # ---------------------------------------
    # Yearly popularity quantile threshold
    # ---------------------------------------
    if "year" in df.columns and "popularity" in df.columns:
        yearly_thresholds = (
            df.groupby("year")["popularity"]
            .quantile(POPULARITY_THRESHOLD / 100)
            .to_dict()
        )

        df["verdict"] = df.apply(
            lambda row: 1
            if row["popularity"] >= yearly_thresholds[row["year"]]
            else 0,
            axis=1,
        )
    else:
        df["verdict"] = 0  # fallback


    # ---------------------------------------
    # Duration quantile features
    # ---------------------------------------
    if "duration_ms" in df.columns:
        Q1 = df["duration_ms"].quantile(0.25)
        Q4 = df["duration_ms"].quantile(0.95)

        df["long_duration"] = df["duration_ms"].apply(lambda x: 1 if x > Q4 else 0)
        df["short_duration"] = df["duration_ms"].apply(lambda x: 1 if x < Q1 else 0)


    # ---------------------------------------
    # Normalizing categorical fields (optional)
    # ---------------------------------------
    if "genre" in df.columns:
        df["genre"] = df["genre"].fillna("unknown")


    # ---------------------------------------
    # DROP UNUSED COLUMNS
    # ---------------------------------------
    drop_cols = [
        "Unnamed: 0",
        "artist_name",
        "track_name",
        "track_id",
        "popularity",
        "year",
        "duration_ms",
    ]

    df = df.drop(columns=drop_cols, errors="ignore")

    return df
