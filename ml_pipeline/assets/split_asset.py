from dagster import asset
import pandas as pd
from sklearn.model_selection import train_test_split

@asset
def split_data(feature_engineered_data: pd.DataFrame):
    """
    Splits the data into training and testing sets.
    Returns a dictionary so Dagster can pass multiple outputs.
    """
    
    df = feature_engineered_data.copy()
    
    # Define target and features
    X = df.drop("verdict", axis=1)
    y = df["verdict"]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }