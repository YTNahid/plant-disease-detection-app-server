"""
Image preprocessing using OpenCV.

Converts raw bytes from an uploaded file into a NumPy array that is
ready to feed directly into the MobileNet model.
"""

import io

import cv2
import numpy as np
from PIL import Image

from core.config import settings
from core.exceptions import InvalidImageException, ImageSizeException

# 10 MB upload limit
MAX_FILE_BYTES = 10 * 1024 * 1024


def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR OpenCV ndarray."""
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidImageException("Could not decode image — unsupported format or corrupted file.")
    return img


def validate_image_size(image_bytes: bytes) -> None:
    """Raise ImageSizeException if the file exceeds MAX_FILE_BYTES."""
    if len(image_bytes) > MAX_FILE_BYTES:
        raise ImageSizeException(
            f"File is too large ({len(image_bytes) / 1024 / 1024:.1f} MB). "
            f"Maximum allowed size is {MAX_FILE_BYTES // 1024 // 1024} MB."
        )


def resize_image(img: np.ndarray, target_size: tuple[int, int] = None) -> np.ndarray:
    """
    Resize the image to `target_size` (width, height) using INTER_AREA for
    downscaling and INTER_LINEAR for upscaling.
    """
    if target_size is None:
        target_size = settings.IMAGE_SIZE

    h, w = img.shape[:2]
    tw, th = target_size

    if (w, h) == (tw, th):
        return img

    interpolation = cv2.INTER_AREA if (w > tw or h > th) else cv2.INTER_LINEAR
    return cv2.resize(img, (tw, th), interpolation=interpolation)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to [0, 1] and convert BGR → RGB.
    Returns a float32 ndarray of shape (H, W, 3).
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb.astype(np.float32) / 255.0


def preprocess(image_bytes: bytes) -> np.ndarray:
    """
    Full preprocessing pipeline.

    Args:
        image_bytes: Raw bytes from an uploaded image file.

    Returns:
        A float32 NumPy array of shape (1, H, W, 3) ready for model.predict().
    """
    validate_image_size(image_bytes)
    img = bytes_to_cv2(image_bytes)
    img = resize_image(img)
    img = normalize_image(img)
    return np.expand_dims(img, axis=0)  # Add batch dimension → (1, 224, 224, 3)


def get_image_info(image_bytes: bytes) -> dict:
    """Return basic metadata about the uploaded image (for debug/logging)."""
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        return {
            "format": pil_img.format,
            "mode": pil_img.mode,
            "size": pil_img.size,
            "file_size_kb": round(len(image_bytes) / 1024, 2),
        }
    except Exception:
        return {"file_size_kb": round(len(image_bytes) / 1024, 2)}