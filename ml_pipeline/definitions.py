"""Dagster definitions for pipeline orchestration."""

from dagster import (
    Definitions,
    load_assets_from_modules,
    define_asset_job,
    ScheduleDefinition,
    build_schedule_context,
)
from dagster_webui import dagit_server

from ml_pipeline import assets
from ml_pipeline.resources.mlflow_resource import mlflow_resource
from ml_pipeline.io_managers.custom_io_manager import local_file_io_manager


# Load all assets
all_assets = load_assets_from_modules([assets.data, assets.features, assets.model])

# Define the complete ML pipeline job
ml_pipeline_job = define_asset_job(
    name="ml_pipeline_job",
    selection=all_assets,
)

# Create definitions
defs = Definitions(
    assets=all_assets,
    jobs=[ml_pipeline_job],
    resources={
        "io_manager": local_file_io_manager.configured(
            {"base_path": "data/processed"}
        ),
        "mlflow": mlflow_resource,
    },
)