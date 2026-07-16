from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from app.services.local_retrieval import LocalRetrievalEngine
from app.core.model_manager import ModelManager
from app.api.deps import get_local_retrieval_engine

router = APIRouter()


class HealthResponse(BaseModel):
    status: str = Field(..., description="System health status (healthy/unhealthy)")
    backend: str = Field(..., description="Backend service status")
    database: str = Field(..., description="Local database availability status")
    embeddings: str = Field(..., description="Embeddings resident status")
    openai: str = Field(..., description="OpenAI API connectivity status")
    version: str = Field(..., description="Application version")


@router.get("/health", response_model=HealthResponse, tags=["System Health"])
async def health_check(
    local_engine: LocalRetrievalEngine = Depends(get_local_retrieval_engine)
) -> HealthResponse:
    """
    Returns system status and check information.
    """
    # Check database status
    db_status = "connected" if local_engine.movies_df is not None else "disconnected"
    
    # Check embeddings status
    emb_status = "loaded" if local_engine.embeddings_matrix is not None else "missing"
    
    # Check OpenAI status
    openai_status = "available"
    try:
        client = ModelManager.get_openai_client()
        if client is None:
            openai_status = "unavailable"
    except Exception:
        openai_status = "unavailable"
        
    overall_status = "healthy"
    if db_status != "connected" or emb_status != "loaded" or openai_status != "available":
        overall_status = "unhealthy"

    return HealthResponse(
        status=overall_status,
        backend="online",
        database=db_status,
        embeddings=emb_status,
        openai=openai_status,
        version="1.0.0"
    )
