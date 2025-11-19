# Setup
uv venv
source .venv/bin/activate
uv sync

# Code quality
ruff format ml_pipeline/
ruff check ml_pipeline/ --fix
pyright ml_pipeline/

# Run pipeline locally
mlflow ui --host 0.0.0.0 --port 5000
dagster dev

# Run API
python -m uvicorn api.main:app --reload --port 8000

# Testing
pytest tests/
pytest --cov=ml_pipeline tests/