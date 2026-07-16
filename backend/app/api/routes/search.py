import time
from typing import List, Optional
from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel, Field
from loguru import logger

from app.services.search_service import SearchService
from app.services.tmdb_service import TMDbService
from app.services.enrichment_helper import enrich_movie_list
from app.api.deps import get_search_service, get_tmdb_service

router = APIRouter()


class SearchResultMovie(BaseModel):
    """Schema representing a single search result with complete metadata, scores, and trailer."""
    id: str = Field(..., description="Unique movie ID string")
    title: str = Field(..., description="Primary movie title")
    original_title: Optional[str] = Field(None, description="Original movie title")
    overview: Optional[str] = Field(None, description="Short synopsis or overview")
    genres: List[str] = Field(default_factory=list, description="List of movie genres")
    directors: List[str] = Field(default_factory=list, description="List of director names")
    cast: List[str] = Field(default_factory=list, description="List of cast member names")
    release_year: Optional[int] = Field(None, description="Release year of the movie")
    rating_value: Optional[float] = Field(None, description="Scaled movie rating in range [1.0, 10.0]")
    popularity: Optional[float] = Field(None, description="TMDb popularity score")
    vote_count: Optional[int] = Field(None, description="Total vote count")
    poster_path: Optional[str] = Field(None, description="TMDb poster image URL suffix path")
    backdrop_path: Optional[str] = Field(None, description="TMDb backdrop image URL suffix path")
    trailer_url: Optional[str] = Field(None, description="YouTube trailer URL link")
    streaming_providers: List[str] = Field(default_factory=list, description="Streaming providers in the US")
    certification: Optional[str] = Field(None, description="US age certification")
    runtime_minutes: Optional[int] = Field(None, description="Runtime in minutes")
    semantic_score: float = Field(..., description="Raw semantic cosine similarity score")
    reranked_score: float = Field(..., description="Hybrid score combined with metadata weights")


class SearchResponse(BaseModel):
    """Schema representing the complete semantic search response with statistics."""
    success: bool = Field(..., description="Indicates request success status")
    message: str = Field(..., description="Descriptive status message")
    query: str = Field(..., description="The original search query string")
    results: List[SearchResultMovie] = Field(..., description="Top matching reranked movie recommendations")
    metadata: dict = Field(..., description="Execution statistics and pagination info")


@router.get("/search", response_model=SearchResponse, tags=["Search"])
async def search(
    q: str = Query(..., min_length=1, description="The query to search for movies semantically"),
    limit: int = Query(10, ge=1, le=100, description="The maximum number of search results to return"),
    search_service: SearchService = Depends(get_search_service),
    tmdb_service: TMDbService = Depends(get_tmdb_service)
) -> SearchResponse:
    """
    Exposes semantic search endpoint. Generates query embeddings, performs nearest-neighbor vector retrieval
    locally, reranks results based on metadata authority, enriches results in real-time using TMDb API,
    and returns ordered recommendations with execution statistics.
    """
    logger.info(f"FastAPI search endpoint hit: q='{q}', limit={limit}")
    start_time = time.perf_counter()
    
    try:
        # 1. Fetch search results from vector database + hybrid reranking
        results = await search_service.search_movies(query=q, limit=limit)
        
        # Determine source by checking cache hits
        source = "database"
        if results and tmdb_service.api_key:
            source = "cache"
            for movie in results:
                tmdb_id = movie.get("tmdb_id")
                if tmdb_id:
                    cached = tmdb_service.cache.get_movie_details(int(tmdb_id))
                    if cached is None:
                        source = "api"  # At least one result triggered API query
                        break

        # 2. Enrich results in real-time concurrently
        enriched_results = await enrich_movie_list(results, tmdb_service)
        
        # 3. Format output
        formatted_results = [SearchResultMovie(**movie) for movie in enriched_results]
        
        elapsed_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        return SearchResponse(
            success=True,
            message="Search results retrieved successfully.",
            query=q,
            results=formatted_results,
            metadata={
                "pagination": {
                    "page": 1,
                    "limit": limit,
                    "total_results": len(formatted_results)
                },
                "execution_statistics": {
                    "elapsed_time_ms": elapsed_time_ms,
                    "source": source
                }
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error in FastAPI search endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while executing the semantic search: {str(e)}"
        )
