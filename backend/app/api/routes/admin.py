from fastapi import APIRouter
from loguru import logger

router = APIRouter()

@router.post("/admin/ingest", tags=["Admin"])
async def trigger_ingestion():
    """
    Triggers the movie data preprocessing and ingestion pipeline.
    """
    logger.warning("Admin trigger: Ingestion pipeline requested.")
    return {
        "status": "triggered",
        "message": "Ingestion pipeline initialization started (placeholder)."
    }

@router.post("/admin/collection/recreate", tags=["Admin"])
async def recreate_collection():
    """
    Recreates the Qdrant vector database collection.
    """
    logger.warning("Admin trigger: Collection recreation requested.")
    return {
        "status": "success",
        "message": "Qdrant collection recreation placeholder executed."
    }
