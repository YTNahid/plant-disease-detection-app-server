import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from core.config import settings
from core.exceptions import register_exception_handlers
from services.inference import load_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)


async def load_model_background():
    """Load model in a background thread so it doesn't block port binding."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_model)
    logger.info("Model loaded successfully.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start model loading in background — port binds immediately
    asyncio.create_task(load_model_background())
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Image classification API using MobileNet",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)
app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": f"{settings.APP_NAME} is running"}


@app.get("/health", tags=["Health"])
async def health_check():
    from services.inference import model_manager
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "model_loaded": model_manager.is_loaded,   # tells you if model is ready
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )