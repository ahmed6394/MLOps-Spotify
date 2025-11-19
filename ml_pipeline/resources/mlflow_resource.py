"""MLflow resource for experiment tracking."""

import mlflow
from dagster import resource


@resource
def mlflow_resource(context):
    """Resource for MLflow tracking server.
    
    Yields:
        MLflow client configured for tracking.
    """
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("spotify-ml-pipeline")
    yield mlflow