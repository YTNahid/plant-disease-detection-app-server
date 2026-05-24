import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter

from core.config import settings
from core.exceptions import ModelNotLoadedException, InferenceException, AppError
from utils.class_names import get_class_name, CLASS_NAMES

logger = logging.getLogger(__name__)

class ModelManager:
    _instance: Optional["ModelManager"] = None
    _interpreter: Optional[Interpreter] = None          # fixed type hint
    _input_details = None
    _output_details = None

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, model_path: str = None) -> None:
        path = model_path or settings.MODEL_PATH
        if not Path(path).exists():
            raise ModelNotLoadedException(f"Model file not found at: {path}")

        logger.info("Loading TFLite model from %s …", path)
        t0 = time.perf_counter()

        self._interpreter = Interpreter(model_path=str(path))
        self._interpreter.allocate_tensors()

        self._input_details  = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        elapsed = time.perf_counter() - t0
        logger.info(
            "TFLite model loaded in %.2f s — input shape: %s",
            elapsed,
            self._input_details[0]["shape"],
        )

    @property
    def is_loaded(self) -> bool:
        return self._interpreter is not None

    def predict(self, input_array: np.ndarray) -> np.ndarray:
        if not self.is_loaded:
            raise ModelNotLoadedException()
        try:
            self._interpreter.set_tensor(
                self._input_details[0]["index"],
                input_array.astype(np.float32),
            )
            self._interpreter.invoke()
            return self._interpreter.get_tensor(
                self._output_details[0]["index"]
            )[0]
        except Exception as exc:
            logger.exception("TFLite inference failed")
            raise InferenceException(str(exc)) from exc

    def format_predictions(
        self,
        raw_probs: np.ndarray,
        top_k: int = None,
    ) -> list[dict]:
        top_k = top_k or settings.TOP_K_RESULTS

        results = [
            {
                "class_index": int(i),
                "class_name": get_class_name(i),
                "confidence": float(raw_probs[i]),
            }
            for i in range(len(raw_probs))   # ← no threshold filter here
        ]
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:top_k]

    def get_model_info(self) -> dict:
        if not self.is_loaded:
            return {"loaded": False}
        input_shape  = self._input_details[0]["shape"].tolist()
        output_shape = self._output_details[0]["shape"].tolist()
        return {
            "loaded": True,
            "input_shape": str(input_shape),
            "output_shape": str(output_shape),
            "num_classes": len(CLASS_NAMES),
            "total_params": "N/A (TFLite model)",
        }


model_manager = ModelManager()


def load_model() -> None:
    model_manager.load()


def run_inference(input_array: np.ndarray, top_k: int = None) -> list[dict]:
    raw_probs = model_manager.predict(input_array)
    predictions = model_manager.format_predictions(raw_probs, top_k=top_k)
    return predictions