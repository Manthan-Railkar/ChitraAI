from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from app.core.config import settings
from app.services.local_retrieval import LocalRetrievalEngine
from app.api.deps import get_local_retrieval_engine

router = APIRouter()


class CacheStatus(BaseModel):
    tmdb_cache: str = Field(..., description="TMDb metadata caching status")


class StatsData(BaseModel):
    total_movies: int = Field(..., description="Total number of loaded canonical movies")
    total_embeddings: int = Field(..., description="Total number of loaded vector embeddings")
    total_genres: int = Field(..., description="Total number of unique movie genres")
    supported_languages: List[str] = Field(..., description="List of supported languages")
    embedding_model: str = Field(..., description="Active SentenceTransformer embedding model name")
    retrieval_model: str = Field(..., description="Retrieval engine identifier")
    backend_version: str = Field(..., description="Backend system version")
    cache_status: CacheStatus = Field(..., description="Cache connection and status info")


class StatsResponse(BaseModel):
    success: bool = Field(..., description="Indicates request success status")
    data: StatsData = Field(..., description="System statistics payload")


@router.get("/stats", response_model=StatsResponse, tags=["System Statistics"])
async def get_stats(
    local_engine: LocalRetrievalEngine = Depends(get_local_retrieval_engine)
) -> StatsResponse:
    """
    Returns statistics and metadata about the local movie database and models.
    """
    if local_engine.movies_df is None:
        local_engine.initialize()

    df = local_engine.movies_df

    # Calculate total genres
    total_genres = 0
    if df is not None:
        try:
            if "genres" in df.columns:
                total_genres = df.select("genres").explode("genres").drop_nulls().unique().height
        except Exception:
            pass

    # Extract unique languages
    languages = ["en"]
    if df is not None:
        try:
            if "language" in df.columns:
                languages = [l for l in df.select("language").drop_nulls().unique().to_series().to_list() if l]
        except Exception:
            pass

    total_movies = df.height if df is not None else 0
    total_embeddings = local_engine.embeddings_matrix.shape[0] if local_engine.embeddings_matrix is not None else 0

    return StatsResponse(
        success=True,
        data=StatsData(
            total_movies=total_movies,
            total_embeddings=total_embeddings,
            total_genres=total_genres,
            supported_languages=languages,
            embedding_model=settings.EMBEDDING_MODEL,
            retrieval_model="Polars Local Retrieval Engine (Cosine Similarity)",
            backend_version="1.0.0",
            cache_status=CacheStatus(tmdb_cache="enabled")
        )
    )
