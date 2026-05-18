from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse


# ── Custom exception types ────────────────────────────────────────────────────

class AppBaseException(Exception):
    """Base class for all application exceptions."""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class ModelNotLoadedException(AppBaseException):
    def __init__(self, message: str = "ML model is not loaded."):
        super().__init__(message, status_code=503)


class InvalidImageException(AppBaseException):
    def __init__(self, message: str = "Uploaded file is not a valid image."):
        super().__init__(message, status_code=422)


class ImageSizeException(AppBaseException):
    def __init__(self, message: str = "Image size is too large."):
        super().__init__(message, status_code=413)


class InferenceException(AppBaseException):
    def __init__(self, message: str = "Error during model inference."):
        super().__init__(message, status_code=500)


# ── Handler registration ──────────────────────────────────────────────────────

def register_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(AppBaseException)
    async def app_exception_handler(request: Request, exc: AppBaseException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.__class__.__name__, "detail": exc.message},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "InternalServerError", "detail": str(exc)},
        )