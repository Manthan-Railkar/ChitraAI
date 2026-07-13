import time
from typing import List, Optional
from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel, Field
from loguru import logger
from qdrant_client.models import Filter, FieldCondition, MatchText

from app.vector_db.qdrant import QdrantWrapper
from app.services.tmdb_service import TMDbService
from app.services.enrichment_helper import enrich_movie_with_tmdb
from app.api.deps import get_qdrant_wrapper, get_tmdb_service

router = APIRouter()


class AutocompleteSuggestion(BaseModel):
    """Schema representing an autocomplete suggestion."""
    id: str = Field(..., description="Unique Qdrant point UUID string")
    title: str = Field(..., description="Movie title")
    release_year: Optional[int] = Field(None, description="Release year of the movie")


class AutocompleteResponse(BaseModel):
    """Schema representing autocomplete responses."""
    query: str = Field(..., description="The autocomplete query string prefix")
    suggestions: List[AutocompleteSuggestion] = Field(..., description="List of matching autocomplete suggestions")


class MovieDetailResponse(BaseModel):
    """Schema representing detailed movie response containing complete metadata, poster, backdrop, and trailer URL."""
    id: str = Field(..., description="Unique Qdrant point UUID string")
    title: str = Field(..., description="Primary movie title")
    original_title: Optional[str] = Field(None, description="Original movie title")
    overview: Optional[str] = Field(None, description="Short synopsis or overview")
    genres: List[str] = Field(default_factory=list, description="List of movie genres")
    directors: List[str] = Field(default_factory=list, description="List of director names")
    cast: List[str] = Field(default_factory=list, description="List of cast member names")
    release_year: Optional[int] = Field(None, description="Release year of the movie")
    rating_value: Optional[float] = Field(None, description="Average rating in range [1.0, 10.0]")
    popularity: Optional[float] = Field(None, description="TMDb popularity score")
    vote_count: Optional[int] = Field(None, description="Total vote count")
    poster_path: Optional[str] = Field(None, description="TMDb poster image URL suffix path")
    backdrop_path: Optional[str] = Field(None, description="TMDb backdrop image URL suffix path")
    trailer_url: Optional[str] = Field(None, description="YouTube trailer URL link")
    streaming_providers: List[str] = Field(default_factory=list, description="Streaming providers in the US")
    certification: Optional[str] = Field(None, description="US age certification")
    runtime_minutes: Optional[int] = Field(None, description="Runtime in minutes")


class MovieEnvelopeResponse(BaseModel):
    """Schema representing wrapped movie details envelope response containing execution statistics."""
    execution_statistics: dict = Field(..., description="Query execution statistics")
    movie: MovieDetailResponse = Field(..., description="Detailed movie record")


@router.get("/movies/autocomplete", response_model=AutocompleteResponse, tags=["Movie"])
async def autocomplete_movies(
    q: str = Query(..., min_length=1, description="The prefix or title substring to autocomplete"),
    limit: int = Query(10, ge=1, le=50, description="The maximum number of autocomplete suggestions to return"),
    qdrant_service: QdrantWrapper = Depends(get_qdrant_wrapper)
) -> AutocompleteResponse:
    """
    Returns autocomplete suggestions for movie titles matching the query substring.
    Uses Qdrant's payload scroll and case-insensitive MatchText match filters.
    """
    logger.info(f"API Movie autocomplete request: q='{q}', limit={limit}")
    
    if not qdrant_service.client:
        qdrant_service.connect()

    try:
        # Query Qdrant scroll with prefix/substring matching filter on title
        response, _ = qdrant_service.client.scroll(
            collection_name=qdrant_service.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="title",
                        match=MatchText(text=q)
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        
        suggestions = []
        for point in response:
            payload = point.payload or {}
            suggestions.append(
                AutocompleteSuggestion(
                    id=str(point.id),
                    title=payload.get("title", "Unknown Movie"),
                    release_year=payload.get("release_year")
                )
            )
            
        return AutocompleteResponse(query=q, suggestions=suggestions)
    except Exception as e:
        logger.error(f"Error executing Qdrant autocomplete: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while executing title autocomplete: {str(e)}"
        )


@router.get("/movies/{movie_id}", response_model=MovieEnvelopeResponse, tags=["Movie"])
async def get_movie_details(
    movie_id: str,
    qdrant_service: QdrantWrapper = Depends(get_qdrant_wrapper),
    tmdb_service: TMDbService = Depends(get_tmdb_service)
) -> MovieEnvelopeResponse:
    """
    Fetches detailed metadata for a specific movie by its Qdrant UUID.
    Retrieves the payload from Qdrant, enriches it in real-time with TMDb API,
    and returns it packaged with query execution statistics.
    """
    logger.info(f"API Movie details request for movie_id='{movie_id}'")
    start_time = time.perf_counter()

    if not qdrant_service.client:
        qdrant_service.connect()

    try:
        points = qdrant_service.client.retrieve(
            collection_name=qdrant_service.collection_name,
            ids=[movie_id]
        )
    except Exception as e:
        logger.error(f"Error fetching movie details from Qdrant: {e}")
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")

    if not points:
        raise HTTPException(status_code=404, detail=f"Movie with ID '{movie_id}' not found.")

    point = points[0]
    payload = point.payload or {}
    
    # Structure basic movie record dict
    movie_dict = {
        "id": str(point.id),
        "title": payload.get("title", "Unknown Movie"),
        "original_title": payload.get("original_title"),
        "overview": payload.get("overview"),
        "genres": payload.get("genres", []),
        "directors": payload.get("directors", []),
        "cast": payload.get("cast", []),
        "release_year": payload.get("release_year"),
        "rating_value": payload.get("rating_value"),
        "popularity": payload.get("popularity"),
        "vote_count": payload.get("vote_count"),
        "poster_path": payload.get("poster_path"),
        "tmdb_id": payload.get("tmdb_id"),
        "imdb_id": payload.get("imdb_id")
    }

    # Determine source by checking cache hit
    source = "database"
    if tmdb_service.api_key and movie_dict.get("tmdb_id"):
        cached = tmdb_service.cache.get_movie_details(int(movie_dict.get("tmdb_id")))
        source = "cache" if cached is not None else "api"

    # Enrich dynamically using TMDb
    enriched_dict = await enrich_movie_with_tmdb(movie_dict, tmdb_service)
    
    # Strip utility fields to conform to response schema
    enriched_dict.pop("tmdb_id", None)
    enriched_dict.pop("imdb_id", None)

    elapsed_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return MovieEnvelopeResponse(
        execution_statistics={
            "elapsed_time_ms": elapsed_time_ms,
            "source": source
        },
        movie=MovieDetailResponse(**enriched_dict)
    )
