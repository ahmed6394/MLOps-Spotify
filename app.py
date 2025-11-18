# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import mlflow
import mlflow.sklearn
import numpy as np
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
expected_features = None
# Use the registered model (recommended) or the specific run
MODEL_URI = "models:/spotify_popularity_predictor/1"  # Using registered model

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info("🔄 Loading MLflow model...")

    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        logger.info(f"Loading model from: {MODEL_URI}")
        model = mlflow.sklearn.load_model(MODEL_URI)
        logger.info("✓ Model loaded successfully!")
        logger.info(f"Model type: {type(model)}")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        try:
            fallback_uri = "runs:/d2b44a62f2364c618af634212357fd8e/model"
            logger.info(f"Trying fallback: {fallback_uri}")
            model = mlflow.sklearn.load_model(fallback_uri)
            logger.info("✓ Model loaded successfully from fallback URI!")
            logger.info(f"Model type: {type(model)}")
        except Exception as fallback_e:
            logger.error(f"✗ Fallback also failed: {fallback_e}")

    # Extract expected feature names
    try:
        global expected_features
        if model is not None:
            # If pipeline with preprocessor
            if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
                prep = model.named_steps["preprocessor"]
                if hasattr(prep, "feature_names_in_"):
                    expected_features = list(prep.feature_names_in_)
            elif hasattr(model, "feature_names_in_"):
                expected_features = list(model.feature_names_in_)
            logger.info(f"Detected feature names: {expected_features}")
    except Exception as fe:
        logger.warning(f"Could not extract feature names: {fe}")

    # Test prediction separated so load success isn't masked
    try:
        if model is not None and expected_features:
            # Provide a generic sample row matching number of features
            sample_values = []
            # Simple heuristic default values
            defaults_map = {
                "danceability": 0.5,
                "energy": 0.5,
                "key": 5,
                "loudness": -10.0,
                "mode": 1,
                "speechiness": 0.05,
                "acousticness": 0.3,
                "instrumentalness": 0.0,
                "liveness": 0.1,
                "valence": 0.4,
                "tempo": 120.0,
                "duration_ms": 180000,
                "time_signature": 4,
                "track_age": 10,
                "year": 2015,
                "track_popularity_hint": 0.5,
                "genre": "pop"
            }
            for f in expected_features:
                if f == "track_age" and "year" in expected_features:
                    sample_values.append(2025 - defaults_map.get("year", 2015))
                else:
                    sample_values.append(defaults_map.get(f, 0))
            sample_df = pd.DataFrame([sample_values], columns=expected_features)
            test_pred = model.predict(sample_df)
            if hasattr(model, "predict_proba"):
                test_prob = model.predict_proba(sample_df)
                logger.info(f"✓ Model test - Prediction: {test_pred[0]}, Prob: {test_prob[0]}")
            else:
                logger.info(f"✓ Model test - Prediction: {test_pred[0]}")
    except Exception as tp_e:
        logger.warning(f"Test prediction failed: {tp_e}")

    yield

    logger.info("🔻 Application shutdown")

app = FastAPI(title="Spotify Predictor", lifespan=lifespan)

@app.get("/")
async def index():
    return {
        "service": "Spotify Predictor",
        "status": "ok",
        "model_loaded": model is not None,
        "feature_count": len(expected_features) if expected_features else 0,
        "endpoints": ["/predict", "/health", "/model-info"],
    }

class TrackFeatures(BaseModel):
    danceability: float
    energy: float
    key: int
    loudness: float
    mode: int
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    duration_ms: int
    time_signature: int
    year: int | None = None
    track_popularity_hint: float | None = None
    genre: str | None = None

@app.post("/predict")
def predict(track: TrackFeatures):
    global model, expected_features
    if model is None:
        return {"error": "Model not loaded", "model_available": False}

    try:
        # Build full map of provided features
        data_map = {
            "danceability": track.danceability,
            "energy": track.energy,
            "key": track.key,
            "loudness": track.loudness,
            "mode": track.mode,
            "speechiness": track.speechiness,
            "acousticness": track.acousticness,
            "instrumentalness": track.instrumentalness,
            "liveness": track.liveness,
            "valence": track.valence,
            "tempo": track.tempo,
            "duration_ms": track.duration_ms,
            "time_signature": track.time_signature,
            "year": track.year,
            "track_popularity_hint": track.track_popularity_hint,
            "genre": track.genre
        }

        # Derived feature: track_age
        if expected_features and "track_age" in expected_features:
            if track.year is None:
                return {"error": "year is required to derive track_age", "model_loaded": True}
            data_map["track_age"] = 2025 - track.year

        # Validate required categorical features
        if expected_features and "genre" in expected_features and track.genre is None:
            return {"error": "genre is required", "model_loaded": True}

        import pandas as pd
        if expected_features:
            row_values = []
            for f in expected_features:
                if f not in data_map:
                    # If missing, default to 0
                    row_values.append(0)
                else:
                    row_values.append(data_map[f])
            df = pd.DataFrame([row_values], columns=expected_features)
        else:
            df = pd.DataFrame([data_map])

        prediction = model.predict(df)[0]

        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(df)[0]
            confidence = float(max(prob))
            probabilities = {
                "not_popular": float(prob[0]),
                "popular": float(prob[1])
            }
        else:
            confidence = 1.0
            probabilities = {"not_popular": 0.5, "popular": 0.5}

        return {
            "prediction": int(prediction),
            "label": "Popular" if prediction == 1 else "Not Popular",
            "confidence": confidence,
            "probabilities": probabilities,
            "model_loaded": True,
            "features_order_used": expected_features
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}", "model_loaded": True}

@app.get("/health")
def health():
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "model_type": str(type(model)) if model else None
    }

@app.get("/model-info")
def model_info():
    if model is None:
        return {"error": "Model not loaded"}

    info = {
        "model_type": str(type(model)),
        "is_pipeline": hasattr(model, 'steps') if model else False,
        "features_used": expected_features if expected_features else None
    }
    if hasattr(model, 'steps'):
        info["pipeline_steps"] = [step[0] for step in model.steps]
    if hasattr(model, 'get_params'):
        info["model_params"] = model.get_params()
    return info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)