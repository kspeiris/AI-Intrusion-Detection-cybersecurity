import json

import joblib
import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import MODEL_METADATA_PATH, RAW_FEATURE_COUNT, SCALER_PATH
from logging_utils import get_logger

logger = get_logger("ids.api")

runtime_state = {
    "loaded": False,
    "error": None,
    "metadata": None,
    "model": None,
    "scaler": None,
    "selector": None,
    "threshold": None,
}


def load_runtime_state():
    with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    runtime_state["metadata"] = metadata
    runtime_state["model"] = joblib.load(metadata["best_model_path"])
    runtime_state["scaler"] = joblib.load(SCALER_PATH)
    runtime_state["selector"] = joblib.load(metadata["best_selector_path"])
    runtime_state["threshold"] = metadata["best_threshold"]
    runtime_state["loaded"] = True
    runtime_state["error"] = None
    logger.info(
        "runtime artifacts loaded",
        extra={"model_name": metadata.get("best_model_name"), "threshold": metadata.get("best_threshold")},
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_runtime_state()
    except Exception as exc:
        runtime_state["loaded"] = False
        runtime_state["error"] = str(exc)
        logger.exception("runtime startup failed")
    yield


app = FastAPI(title="AI-Based Intrusion Detection System", lifespan=lifespan)


class NetworkTraffic(BaseModel):
    features: list[float] = Field(..., min_length=RAW_FEATURE_COUNT, max_length=RAW_FEATURE_COUNT)


def ensure_runtime_ready():
    if not runtime_state["loaded"]:
        logger.warning("prediction requested while runtime unavailable")
        raise HTTPException(
            status_code=503,
            detail=runtime_state["error"] or "Model artifacts are not available.",
        )


@app.get("/")
def home():
    metadata = runtime_state["metadata"] or {}
    return {
        "message": "AI-Based Intrusion Detection System is running" if runtime_state["loaded"] else "AI-Based Intrusion Detection System is unavailable",
        "loaded": runtime_state["loaded"],
        "model": metadata.get("best_model_name"),
        "threshold": runtime_state["threshold"],
        "error": runtime_state["error"],
    }


@app.get("/health")
def health():
    return {
        "loaded": runtime_state["loaded"],
        "error": runtime_state["error"],
    }


@app.post("/predict")
def predict(data: NetworkTraffic):
    ensure_runtime_ready()

    try:
        features = np.array(data.features, dtype=float).reshape(1, -1)
        features = runtime_state["scaler"].transform(features)
        features = runtime_state["selector"].transform(features)
        probability = float(runtime_state["model"].predict_proba(features)[0, 1])
        prediction = int(probability >= runtime_state["threshold"])
        logger.info(
            "prediction completed",
            extra={
                "prediction": prediction,
                "probability": round(probability, 4),
                "threshold": runtime_state["threshold"],
            },
        )
    except ValueError as exc:
        logger.warning(f"invalid prediction payload: {exc}")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "threshold": runtime_state["threshold"],
        "result": "Attack" if prediction == 1 else "Normal"
    }
