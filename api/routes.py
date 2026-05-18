# --------- Endpoints ---------
# POST /api/v1/predict        – Upload an image, get classification results.
# POST /api/v1/predict/batch  – Upload multiple images at once.
# GET  /api/v1/model/info     – Model metadata (input shape, class count …).
# GET  /api/v1/classes        – List all class labels.

import time
import logging
from typing import Annotated

from fastapi import APIRouter, File, UploadFile, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.config import settings
from core.exceptions import AppError, InvalidImageException
from services.preprocess import preprocess, get_image_info
from services.inference import run_inference, model_manager
from utils.class_names import get_all_classes, get_disease_info

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Response schemas ──────────────────────────────────────────────────────────

class Prediction(BaseModel):
    class_index: int
    class_name: str
    confidence: float

class TopPrediction(BaseModel):
    class_index: int
    class_name: str
    confidence: float
    description: str | None = None
    treatment: str | None = None
    prevention: str | None = None

class PredictResponse(BaseModel):
    filename: str
    predictions: list[Prediction]
    top_prediction: TopPrediction | None = None
    inference_time_ms: float
    image_info: dict
    error: str | None = None
    best_confidence: str | None = None

class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    total_images: int
    total_time_ms: float

class ModelInfoResponse(BaseModel):
    loaded: bool
    input_shape: str | None = None
    output_shape: str | None = None
    num_classes: int | None = None
    total_params: int | None = None


# ── Allowed MIME types ────────────────────────────────────────────────────────

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}

def _validate_upload(upload: UploadFile) -> None:
    if upload.content_type not in ALLOWED_TYPES:
        raise InvalidImageException(
            f"Unsupported content type '{upload.content_type}'. "
            f"Allowed types: {', '.join(sorted(ALLOWED_TYPES))}"
        )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classify a single image",
    tags=["Inference"],
)
async def predict(
    file: Annotated[UploadFile, File(description="Image file to classify")],
    top_k: int = Query(default=None, ge=1, le=20, description="Number of top results to return"),
):
    _validate_upload(file)
    image_bytes = await file.read()
    img_info = get_image_info(image_bytes)
    logger.info("Received image '%s' (%s KB)", file.filename, img_info.get("file_size_kb"))

    t0 = time.perf_counter()

    try:
        input_array = preprocess(image_bytes)
        predictions = run_inference(input_array, top_k=top_k or settings.TOP_K_RESULTS)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        top_pred_raw = predictions[0]
        top_info = get_disease_info(top_pred_raw["class_name"])
        detailed_top_prediction = TopPrediction(**top_pred_raw, **top_info)

        return PredictResponse(
            filename=file.filename or "unknown",
            predictions=[Prediction(**p) for p in predictions],
            top_prediction=detailed_top_prediction,
            inference_time_ms=round(elapsed_ms, 2),
            image_info=img_info,
            error=None,
            best_confidence=None,
        )

    except AppError as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return PredictResponse(
            filename=file.filename or "unknown",
            predictions=[],
            top_prediction=None,
            inference_time_ms=round(elapsed_ms, 2),
            image_info=img_info,
            error=e.message,
            best_confidence=getattr(e, "best_confidence", None),
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Classify multiple images",
    tags=["Inference"],
)
async def predict_batch(
    files: Annotated[list[UploadFile], File(description="Image files to classify")],
    top_k: int = Query(default=None, ge=1, le=20),
):
    if len(files) > settings.MAX_IMAGE_UPLOAD:
        raise AppError(f"Maximum {settings.MAX_IMAGE_UPLOAD} images per batch request.", 400)

    t_batch_start = time.perf_counter()
    results: list[PredictResponse] = []

    for upload in files:
        t0 = time.perf_counter()
        img_info = {}

        try:
            _validate_upload(upload)
            image_bytes = await upload.read()
            img_info = get_image_info(image_bytes)

            input_array = preprocess(image_bytes)
            predictions = run_inference(input_array, top_k=top_k or settings.TOP_K_RESULTS)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            top_pred_raw = predictions[0]
            top_info = get_disease_info(top_pred_raw["class_name"])
            detailed_top = TopPrediction(**top_pred_raw, **top_info)

            results.append(PredictResponse(
                filename=upload.filename or "unknown",
                predictions=[Prediction(**p) for p in predictions],
                top_prediction=detailed_top,
                inference_time_ms=round(elapsed_ms, 2),
                image_info=img_info,
                error=None,
            ))

        except AppError as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            results.append(PredictResponse(
                filename=upload.filename or "unknown",
                predictions=[],
                top_prediction=None,
                inference_time_ms=round(elapsed_ms, 2),
                image_info=img_info,
                error=e.message,
                best_confidence=getattr(e, "best_confidence", None),  # ← add this
            ))

    total_ms = (time.perf_counter() - t_batch_start) * 1000
    return BatchPredictResponse(
        results=results,
        total_images=len(results),
        total_time_ms=round(total_ms, 2),
    )


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Get model metadata",
    tags=["Model"],
)
async def model_info():
    return ModelInfoResponse(**model_manager.get_model_info())


@router.get(
    "/classes",
    summary="List all class labels",
    tags=["Model"],
)
async def list_classes():
    return JSONResponse(content={"classes": get_all_classes()})