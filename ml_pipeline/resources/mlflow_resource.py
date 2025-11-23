from dagster import resource
import mlflow


@resource
def mlflow_resource(_):
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("spotify_tracks_experiment")

    return mlflow