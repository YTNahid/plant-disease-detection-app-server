from pathlib import Path
from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Plant Disease Classifier using MobileNetV2"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS
    ALLOWED_ORIGINS: list[str] = ["*"]

    # Model
    MODEL_PATH: str = str(BASE_DIR / "model" / "mobilenet_best_model.tflite")

    # Image preprocessing
    IMAGE_SIZE: tuple[int, int] = (224, 224)   # MobileNet default input size
    IMAGE_CHANNELS: int = 5
    MAX_FILE_SIZE_MB: int = 10
    MAX_IMAGE_UPLOAD: int = 10

    # Inference
    TOP_K_RESULTS: int = 5
    CONFIDENCE_THRESHOLD: float = 0.50

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()