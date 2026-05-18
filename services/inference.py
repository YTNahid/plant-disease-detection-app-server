"""
TensorFlow model loader and inference engine.

The model is loaded once at startup (singleton pattern) so that every
request reuses the same in-memory weights rather than reloading from disk.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

from core.config import settings
from core.exceptions import ModelNotLoadedException, InferenceException
from utils.class_names import get_class_name, CLASS_NAMES

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton that holds the loaded Keras model."""

    _instance: Optional["ModelManager"] = None
    _model: Optional[tf.keras.Model] = None

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self, model_path: str = None) -> None:
        """Load (or reload) the .h5 model from disk."""
        path = model_path or settings.MODEL_PATH
        if not Path(path).exists():
            raise ModelNotLoadedException(f"Model file not found at: {path}")

        logger.info("Loading model from %s …", path)
        t0 = time.perf_counter()
        self._model = tf.keras.models.load_model(path, compile=False)
        elapsed = time.perf_counter() - t0
        logger.info("Model loaded in %.2f s — input shape: %s", elapsed, self._model.input_shape)

    @property
    def model(self) -> tf.keras.Model:
        if self._model is None:
            raise ModelNotLoadedException()
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, input_array: np.ndarray) -> np.ndarray:
        """
        Run inference on a pre-processed batch array.

        Args:
            input_array: float32 array of shape (1, H, W, C).

        Returns:
            1-D float32 probability array of shape (num_classes,).
        """
        try:
            predictions = self.model.predict(input_array, verbose=0)
            return predictions[0]          # strip batch dimension
        except Exception as exc:
            logger.exception("Inference failed")
            raise InferenceException(str(exc)) from exc

    # ── Result formatting ─────────────────────────────────────────────────────

    def format_predictions(
        self,
        raw_probs: np.ndarray,
        top_k: int = None,
        threshold: float = None,
    ) -> list[dict]:
        """
        Convert raw probability array → sorted list of prediction dicts.

        Args:
            raw_probs:  1-D probability array from predict().
            top_k:      Return only the top-k results.
            threshold:  Exclude results below this confidence.

        Returns:
            List of {"class_index", "class_name", "confidence"} dicts,
            sorted by confidence descending.
        """
        top_k = top_k or settings.TOP_K_RESULTS
        threshold = threshold if threshold is not None else settings.CONFIDENCE_THRESHOLD

        # Build full result list
        results = [
            {
                "class_index": int(i),
                "class_name": get_class_name(i),
                "confidence": float(raw_probs[i]),
            }
            for i in range(len(raw_probs))
            if raw_probs[i] >= threshold
        ]

        # Sort by confidence and cap at top_k
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:top_k]

    def get_model_info(self) -> dict:
        """Return a summary of the loaded model."""
        if not self.is_loaded:
            return {"loaded": False}
        return {
            "loaded": True,
            "input_shape": str(self._model.input_shape),
            "output_shape": str(self._model.output_shape),
            "num_classes": len(CLASS_NAMES),
            "total_params": self._model.count_params(),
        }


# ── Module-level singleton ────────────────────────────────────────────────────

model_manager = ModelManager()


def load_model() -> None:
    """Called once at app startup to load the model into memory."""
    model_manager.load()


def run_inference(input_array: np.ndarray, top_k: int = None) -> list[dict]:
    """
    Convenience wrapper used by API routes.

    Args:
        input_array: Pre-processed float32 array (1, H, W, C).
        top_k:       How many top predictions to return.

    Returns:
        Sorted list of prediction dicts.
    """
    raw_probs = model_manager.predict(input_array)
    return model_manager.format_predictions(raw_probs, top_k=top_k)