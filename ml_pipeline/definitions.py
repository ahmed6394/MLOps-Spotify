from dagster import Definitions, load_assets_from_package_module
from ml_pipeline import assets
from ml_pipeline.resources.mlflow_resource import mlflow_resource

# Load all assets from the assets/ package
all_assets = load_assets_from_package_module(assets)

# Dagster Definitions
defs = Definitions(
    assets=all_assets,
    resources={
        "mlflow": mlflow_resource
    },
)

