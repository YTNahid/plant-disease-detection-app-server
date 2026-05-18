import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


# ── AppError — base class ─────────────────────────────────────────────────────
class AppError(Exception):
    def __init__(self, message: str, status_code: int = 500, best_confidence: str = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.status = "fail" if str(status_code).startswith("4") else "error"
        self.is_operational = True
        self.best_confidence = best_confidence

# ── Shortcut error classes ────────────────────────────────────────────────────

class NotFoundError(AppError):
    def __init__(self, message="Resource not found"):
        super().__init__(message, 404)

class BadRequestError(AppError):
    def __init__(self, message="Bad request"):
        super().__init__(message, 400)

class UnauthorizedError(AppError):
    def __init__(self, message="Unauthorized, please login again"):
        super().__init__(message, 401)

class ForbiddenError(AppError):
    def __init__(self, message="You do not have permission"):
        super().__init__(message, 403)

class ValidationError(AppError):
    def __init__(self, message="Validation failed"):
        super().__init__(message, 422)

class ServiceUnavailableError(AppError):
    def __init__(self, message="Service unavailable"):
        super().__init__(message, 503)

# ML-specific
class ModelNotLoadedException(AppError):
    def __init__(self, message="ML model is not loaded yet"):
        super().__init__(message, 503)

class InvalidImageException(AppError):
    def __init__(self, message="Invalid or corrupted image file"):
        super().__init__(message, 422)

class ImageSizeException(AppError):
    def __init__(self, message="Image file is too large"):
        super().__init__(message, 413)

class InferenceException(AppError):
    def __init__(self, message="Error during model inference"):
        super().__init__(message, 500)


# ── Exception converters ──────────────────────────────────────────────────────

def handle_validation_error(exc: RequestValidationError) -> AppError:
    errors = [f"{'.'.join(str(l) for l in e['loc'])}: {e['msg']}" for e in exc.errors()]
    message = f"Invalid input data. {'. '.join(errors)}"
    return AppError(message, 422)

def handle_value_error(exc: ValueError) -> AppError:
    return AppError(f"Invalid value: {str(exc)}", 400)

def handle_type_error(exc: TypeError) -> AppError:
    return AppError(f"Type error: {str(exc)}", 400)


# ── Central error dispatcher ──────────────────────────────────────────────────

def handle_error(exc: Exception) -> JSONResponse:
    if isinstance(exc, RequestValidationError):
        exc = handle_validation_error(exc)
    elif isinstance(exc, ValueError):
        exc = handle_value_error(exc)
    elif isinstance(exc, TypeError):
        exc = handle_type_error(exc)
    elif not isinstance(exc, AppError):
        # Unknown/unhandled — log it and return generic message
        logger.error("Unhandled error: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Something went wrong"},
        )

    return JSONResponse(
        status_code=exc.status_code,
        content={"status": exc.status, "message": exc.message},
    )


# ── Register with FastAPI ─────────────────────────────────────────────────────

def register_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        return handle_error(exc)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return handle_error(exc)

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return handle_error(exc)

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        return handle_error(exc)