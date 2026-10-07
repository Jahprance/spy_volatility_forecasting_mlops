from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from app.monitoring import check_input_ranges, load_reference
from app.schema import FEATURE_ORDER, PredictionInput

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "har_ridge_var.joblib"
REFERENCE_PATH = ROOT_DIR / "models" / "har_feature_reference.json"

model = joblib.load(MODEL_PATH)
reference = load_reference(REFERENCE_PATH)

app = FastAPI(
    title="SPY Next-Day Volatility Forecasting MLOps Service",
    description="Predicts next-day SPY Parkinson variance using HAR-Ridge.",
)


def feature_dict(request: PredictionInput) -> dict:
    return {
        feature: float(getattr(request, feature))
        for feature in FEATURE_ORDER
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": True,
        "model_file": MODEL_PATH.name,
    }


@app.post("/predict")
def predict(request: PredictionInput) -> dict:
    features = feature_dict(request)
    feature_frame = pd.DataFrame([features], columns=FEATURE_ORDER)

    raw_prediction = float(model.predict(feature_frame)[0])
    prediction = max(raw_prediction, 0.0)
    monitoring = check_input_ranges(features, reference)

    return {
        "predicted_variance": prediction,
        "monitoring": monitoring,
    }


@app.post("/monitor")
def monitor(request: PredictionInput) -> dict:
    return check_input_ranges(feature_dict(request), reference)
