import io
import cv2
import numpy as np
from PIL import Image

from core.config import settings
from core.exceptions import InvalidImageException, ImageSizeException

import tensorflow as tf
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Get image ready for OpenCV
def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidImageException()
    return img

# Validate image size
def validate_image_size(image_bytes: bytes) -> None:
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise ImageSizeException(
            f"File is too large ({len(image_bytes) / 1024 / 1024:.1f} MB). "
            f"Maximum allowed size is {settings.MAX_FILE_SIZE_MB} MB."
        )

# Resize image to target size.
def resize_image(img: np.ndarray, target_size: tuple[int, int] = None) -> np.ndarray:
    if target_size is None:
        target_size = settings.IMAGE_SIZE

    h, w = img.shape[:2]
    tw, th = target_size

    if (w, h) == (tw, th):
        return img

    interpolation = cv2.INTER_AREA if (w > tw or h > th) else cv2.INTER_LINEAR
    return cv2.resize(img, (tw, th), interpolation=interpolation)

# Normalize image for mobilenetv2
def normalize_image(img: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32)
    return preprocess_input(img_float)       

# Begin image preprocessing pipeline.
def preprocess(image_bytes: bytes) -> np.ndarray:
    validate_image_size(image_bytes)
    img = bytes_to_cv2(image_bytes)       
    img = resize_image(img)          
    img = normalize_image(img)
    return np.expand_dims(img, axis=0)

# Get image metadata
def get_image_info(image_bytes: bytes) -> dict:
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