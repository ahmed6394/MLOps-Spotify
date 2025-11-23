from dagster import asset
import pandas as pd

POPULARITY_THRESHOLD = 85


@asset
def feature_engineered_data(clean_data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds feature engineering:
    - artist_song_count
    - yearly popularity thresholds
    - verdict (binary label)
    - long & short duration features
    """

    df = clean_data.copy()

    # artist-wise song count
    df['artist_song_count'] = df.groupby('artist_name')['track_id'].transform('count')

    # ensure year column is integer
    df["year"] = df["year"].astype(int)

    # yearly popularity quantile
    yearly_thresholds = (
        df.groupby("year")["popularity"]
        .quantile(POPULARITY_THRESHOLD / 100)
        .to_dict()
    )

    # verdict: >= threshold → 1 else 0
    df["verdict"] = df.apply(
        lambda row: 1
        if row["popularity"] >= yearly_thresholds[row["year"]]
        else 0,
        axis=1,
    )

    # duration quantiles
    Q1 = df["duration_ms"].quantile(0.25)
    Q4 = df["duration_ms"].quantile(0.95)

    # duration category
    df["long_duration"] = df["duration_ms"].apply(lambda x: 1 if x > Q4 else 0)
    df["short_duration"] = df["duration_ms"].apply(lambda x: 1 if x < Q1 else 0)

    return df
