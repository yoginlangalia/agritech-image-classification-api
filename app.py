"""
AgriTech Quality Detection API
FastAPI wrapper around existing fruit/vegetable prediction models.
"""
import os
import io
import warnings
import logging
from contextlib import asynccontextmanager

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import xgboost as xgb
from PIL import Image, ImageOps
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── Config ───────────────────────────────────────────────

IMG_SIZE = (224, 224)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    "fruit": {
        "cnn": os.path.join(BASE_DIR, "fruit_mobilenetv2_feature_extractor.h5"),
        "xgb": os.path.join(BASE_DIR, "fruit_quality_xgboost.json"),
    },
    "vegetable": {
        "cnn": os.path.join(BASE_DIR, "vegetable_mobilenetv2_feature_extractor.h5"),
        "xgb": os.path.join(BASE_DIR, "vegetable_quality_xgboost.json"),
    },
}

CLASS_NAMES = {
    "fruit": [
        "fresh_apple", "fresh_banana", "fresh_mango", "fresh_orange", "fresh_strawberry",
        "rotten_apple", "rotten_banana", "rotten_mango", "rotten_orange", "rotten_strawberry",
    ],
    "vegetable": [
        "fresh_bellpepper", "fresh_carrot", "fresh_cucumber", "fresh_potato", "fresh_tomato",
        "rotten_bellpepper", "rotten_carrot", "rotten_cucumber", "rotten_potato", "rotten_tomato",
    ],
}

RATING_THRESHOLDS = {5: 0.95, 4: 0.85, 3: 0.70, 2: 0.55, 1: 0.40}

INTERPRETATIONS = {
    "rotten": "This item is ROTTEN and should NOT be consumed. Discard immediately.",
    0: "Low confidence prediction. Manual inspection required.",
    1: "Poor quality detected. Inspect carefully before consuming.",
    2: "Fair quality - consumable but not optimal. Use soon or in cooked dishes.",
    3: "Good quality - suitable for consumption. Safe to eat within normal timeframe.",
    4: "Very good quality - fresh and healthy. Excellent for eating or selling.",
    5: "Excellent quality - premium grade! Perfect for direct consumption or premium sale.",
}

# ─── Response Schema ──────────────────────────────────────

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    quality_rating: int
    rating_label: str
    star_display: str
    interpretation: str
    all_probabilities: dict[str, float]

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool

# ─── Models holder ────────────────────────────────────────

models = {}

# ─── Core prediction logic (from predict_fruit.py / predict_vegetable.py) ──

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = ImageOps.exif_transpose(img)
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def calculate_rating(predicted_class: str, confidence: float) -> tuple[int, str]:
    if predicted_class.startswith("rotten"):
        return 0, "Rejected - Do Not Consume"
    if confidence >= RATING_THRESHOLDS[5]:
        return 5, "Excellent Quality"
    elif confidence >= RATING_THRESHOLDS[4]:
        return 4, "Very Good Quality"
    elif confidence >= RATING_THRESHOLDS[3]:
        return 3, "Good Quality"
    elif confidence >= RATING_THRESHOLDS[2]:
        return 2, "Fair Quality"
    elif confidence >= RATING_THRESHOLDS[1]:
        return 1, "Poor Quality - Inspect Carefully"
    else:
        return 0, "Uncertain - Manual Inspection Required"


def get_star_display(rating: int) -> str:
    return "\u2605" * rating + "\u2606" * (5 - rating)


def predict(image_bytes: bytes, pipeline: str) -> dict:
    cnn_model = models[pipeline]["cnn"]
    xgb_model = models[pipeline]["xgb"]
    class_names = CLASS_NAMES[pipeline]

    img_array = preprocess_image(image_bytes)
    features = cnn_model.predict(img_array, verbose=0)
    probs = xgb_model.predict_proba(features)[0]

    best_idx = int(np.argmax(probs))
    predicted_class = class_names[best_idx]
    confidence = float(probs[best_idx])

    rating, rating_label = calculate_rating(predicted_class, confidence)

    if predicted_class.startswith("rotten"):
        interpretation = INTERPRETATIONS["rotten"]
    else:
        interpretation = INTERPRETATIONS[rating]

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "quality_rating": rating,
        "rating_label": rating_label,
        "star_display": get_star_display(rating),
        "interpretation": interpretation,
        "all_probabilities": {
            name: float(prob) for name, prob in zip(class_names, probs)
        },
    }

# ─── FastAPI App ──────────────────────────────────────────

logger = logging.getLogger("agritech-api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models...")
    # MobileNetV2 feature extractor — same architecture used during training.
    # Weights were frozen (no fine-tuning), so standard ImageNet weights are identical.
    # Shared across both pipelines since the CNN is the same.
    cnn = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False,
        input_shape=(224, 224, 3), pooling="avg"
    )
    for pipeline in ("fruit", "vegetable"):
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(MODEL_PATHS[pipeline]["xgb"])
        models[pipeline] = {"cnn": cnn, "xgb": xgb_model}
    logger.info("All models loaded.")
    yield
    models.clear()

app = FastAPI(title="AgriTech Quality Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "*")],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


async def handle_prediction(file: UploadFile, pipeline: str) -> PredictionResponse:
    if file.content_type not in ("image/jpeg", "image/png", "image/webp", "image/bmp"):
        raise HTTPException(400, f"Unsupported file type: {file.content_type}. Use JPEG, PNG, WebP, or BMP.")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(400, "Empty file.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large. Maximum 10 MB.")

    try:
        result = predict(image_bytes, pipeline)
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(500, "Prediction failed. Please try a different image.")

    return PredictionResponse(**result)


@app.post("/api/predict/fruit", response_model=PredictionResponse)
async def predict_fruit(file: UploadFile = File(...)):
    return await handle_prediction(file, "fruit")


@app.post("/api/predict/vegetable", response_model=PredictionResponse)
async def predict_vegetable(file: UploadFile = File(...)):
    return await handle_prediction(file, "vegetable")


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if models else "loading",
        models_loaded=bool(models),
    )
