# --------- Endpoints ---------
# POST /api/v1/predict        – Upload an image, get classification results.
# POST /api/v1/predict/batch  – Upload multiple images at once.
# GET  /api/v1/model/info     – Model metadata (input shape, class count …).
# GET  /api/v1/classes        – List all class labels.

import time
import logging
from typing import Annotated

from fastapi import APIRouter, File, UploadFile, HTTPException, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.config import settings
from core.exceptions import InvalidImageException
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
    top_prediction: TopPrediction
    inference_time_ms: float
    image_info: dict


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
    """
    Upload an image and receive the top-k predicted classes with confidence scores.
    """
    _validate_upload(file)
    image_bytes = await file.read()

    img_info = get_image_info(image_bytes)
    logger.info("Received image '%s' (%s KB)", file.filename, img_info.get("file_size_kb"))

    t0 = time.perf_counter()
    input_array = preprocess(image_bytes)
    predictions = run_inference(input_array, top_k=top_k or settings.TOP_K_RESULTS)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if not predictions:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Model returned no predictions above the confidence threshold.",
        )
    
    top_pred_raw = predictions[0]
    top_info = get_disease_info(top_pred_raw["class_name"])
    detailed_top_prediction = TopPrediction(**top_pred_raw, **top_info)

    # 4. Return the response
    return PredictResponse(
        filename=file.filename or "unknown",
        predictions=[Prediction(**p) for p in predictions],
        top_prediction=detailed_top_prediction,           
        inference_time_ms=round(elapsed_ms, 2),
        image_info=img_info,
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
    """
    Upload up to 10 images at once and get predictions for each.
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10 images per batch request.",
        )

    t_batch_start = time.perf_counter()
    results: list[PredictResponse] = []

    for upload in files:
        _validate_upload(upload)
        image_bytes = await upload.read()
        img_info = get_image_info(image_bytes)

        t0 = time.perf_counter()
        input_array = preprocess(image_bytes)
        predictions = run_inference(input_array, top_k=top_k or settings.TOP_K_RESULTS)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # 1. Create the lightweight list for the main array (FIXED: Using Prediction)
        basic_predictions = [Prediction(**p) for p in predictions]

        # 2. Handle the top prediction and fetch extra details
        if predictions:
            top_pred_raw = predictions[0]
            top_info = get_disease_info(top_pred_raw["class_name"])
            # FIXED: Using TopPrediction
            detailed_top = TopPrediction(**top_pred_raw, **top_info)
        else:
            # Fallback if the model returned nothing above the confidence threshold
            # FIXED: Using TopPrediction
            detailed_top = TopPrediction(
                class_index=-1, 
                class_name="none", 
                confidence=0.0,
                description="No predictions available.",
                treatment="N/A",
                prevention="N/A"
            )

        # 3. Append the formatted response
        results.append(
            PredictResponse(
                filename=upload.filename or "unknown",
                predictions=basic_predictions,
                top_prediction=detailed_top,
                inference_time_ms=round(elapsed_ms, 2),
                image_info=img_info,
            )
        )

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
    """Return information about the currently loaded model."""
    return ModelInfoResponse(**model_manager.get_model_info())


@router.get(
    "/classes",
    summary="List all class labels",
    tags=["Model"],
)
async def list_classes():
    """Return a mapping of class index → class name for the loaded model."""
    return JSONResponse(content={"classes": get_all_classes()})