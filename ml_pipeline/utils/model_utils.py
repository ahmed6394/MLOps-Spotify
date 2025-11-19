"""Model training and evaluation utilities."""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from typing import Dict, Any, Tuple
import pickle


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42
) -> RandomForestClassifier:
    """Train Random Forest model.
    
    Args:
        X_train: Training features.
        y_train: Training target.
        n_estimators: Number of trees.
        random_state: Random seed.
        
    Returns:
        Trained RandomForestClassifier.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    random_state: int = 42
) -> XGBClassifier:
    """Train XGBoost model.
    
    Args:
        X_train: Training features.
        y_train: Training target.
        n_estimators: Number of boosting rounds.
        learning_rate: Learning rate.
        random_state: Random seed.
        
    Returns:
        Trained XGBClassifier.
    """
    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """Evaluate model performance.
    
    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test target.
        
    Returns:
        Dictionary with evaluation metrics.
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
    }
    return metrics