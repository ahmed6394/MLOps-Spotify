"""Model training and evaluation assets."""

import mlflow
from dagster import asset, get_dagster_logger
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd
from ml_pipeline.utils.model_utils import (
    train_random_forest,
    train_xgboost,
    evaluate_model,
)


@asset
def random_forest_model(train_features_and_target: tuple):
    """Train Random Forest model.
    
    Args:
        train_features_and_target: Tuple of (X_train, y_train).
        
    Returns:
        Trained Random Forest model.
    """
    logger = get_dagster_logger()
    X_train, y_train = train_features_and_target
    
    logger.info("Training Random Forest model...")
    
    with mlflow.start_run(run_name="random_forest"):
        model = train_random_forest(X_train, y_train, n_estimators=100)
        mlflow.sklearn.log_model(model, "model")
        logger.info("Random Forest model trained and logged")
    
    return model


@asset
def xgboost_model(train_features_and_target: tuple):
    """Train XGBoost model.
    
    Args:
        train_features_and_target: Tuple of (X_train, y_train).
        
    Returns:
        Trained XGBoost model.
    """
    logger = get_dagster_logger()
    X_train, y_train = train_features_and_target
    
    logger.info("Training XGBoost model...")
    
    with mlflow.start_run(run_name="xgboost"):
        model = train_xgboost(X_train, y_train, n_estimators=100)
        mlflow.xgboost.log_model(model, "model")
        logger.info("XGBoost model trained and logged")
    
    return model


@asset
def random_forest_metrics(
    random_forest_model: RandomForestClassifier,
    test_features_and_target: tuple
) -> dict:
    """Evaluate Random Forest model.
    
    Args:
        random_forest_model: Trained RF model.
        test_features_and_target: Tuple of (X_test, y_test).
        
    Returns:
        Evaluation metrics dictionary.
    """
    logger = get_dagster_logger()
    X_test, y_test = test_features_and_target
    
    logger.info("Evaluating Random Forest model...")
    metrics = evaluate_model(random_forest_model, X_test, y_test)
    
    with mlflow.start_run():
        mlflow.log_metrics(metrics)
    
    logger.info(f"Random Forest metrics: {metrics}")
    return metrics


@asset
def xgboost_metrics(
    xgboost_model: XGBClassifier,
    test_features_and_target: tuple
) -> dict:
    """Evaluate XGBoost model.
    
    Args:
        xgboost_model: Trained XGBoost model.
        test_features_and_target: Tuple of (X_test, y_test).
        
    Returns:
        Evaluation metrics dictionary.
    """
    logger = get_dagster_logger()
    X_test, y_test = test_features_and_target
    
    logger.info("Evaluating XGBoost model...")
    metrics = evaluate_model(xgboost_model, X_test, y_test)
    
    with mlflow.start_run():
        mlflow.log_metrics(metrics)
    
    logger.info(f"XGBoost metrics: {metrics}")
    return metrics