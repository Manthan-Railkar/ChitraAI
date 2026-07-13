from fastapi import APIRouter
from loguru import logger
from app.core.config import settings

router = APIRouter()

@router.get("/health", tags=["System Health"])
async def health_check():
    """
    Returns system status and configuration information.
    """
    logger.debug("Health check endpoint hit.")
    return {
        "status": "healthy",
        "version": "1.0.0",
        "config": {
            "qdrant_url": settings.QDRANT_URL,
            "qdrant_collection": settings.QDRANT_COLLECTION,
            "embedding_model": settings.EMBEDDING_MODEL,
            "device": settings.DEVICE,
            "log_level": settings.LOG_LEVEL
        }
    }
