"""Custom IO manager for data persistence."""

import pandas as pd
import pickle
from pathlib import Path
from dagster import IOManager, io_manager
from typing import Any


class LocalFileIOManager(IOManager):
    """IO manager that stores/loads data from local filesystem."""
    
    def __init__(self, base_path: str) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def handle_output(self, context, obj: Any) -> None:
        """Save output to disk.
        
        Args:
            context: Dagster context.
            obj: Object to save.
        """
        output_path = self.base_path / f"{context.step_key}.pkl"
        
        if isinstance(obj, pd.DataFrame):
            obj.to_parquet(output_path.with_suffix(".parquet"))
        else:
            with open(output_path, "wb") as f:
                pickle.dump(obj, f)
        
        context.log.info(f"Saved output to {output_path}")
    
    def load_input(self, context) -> Any:
        """Load input from disk.
        
        Args:
            context: Dagster context.
            
        Returns:
            Loaded object.
        """
        parent_step_key = context.upstream_step_key
        input_path = self.base_path / f"{parent_step_key}.pkl"
        
        if input_path.with_suffix(".parquet").exists():
            return pd.read_parquet(input_path.with_suffix(".parquet"))
        
        with open(input_path, "rb") as f:
            return pickle.load(f)


@io_manager(config_schema={"base_path": str})
def local_file_io_manager(context):
    """Create LocalFileIOManager.
    
    Args:
        context: Dagster context.
        
    Returns:
        LocalFileIOManager instance.
    """
    return LocalFileIOManager(context.resource_config["base_path"])