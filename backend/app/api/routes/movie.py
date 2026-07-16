import time
import polars as pl
import threading
from collections import OrderedDict
from typing import List, Optional, Any, Dict
from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel, Field
from loguru import logger

from app.core.config import settings
from app.services.local_retrieval import LocalRetrievalEngine
from app.services.tmdb_service import TMDbService
from app.services.enrichment_helper import enrich_movie_with_tmdb
from app.api.deps import get_local_retrieval_engine, get_tmdb_service

router = APIRouter()


class AutocompleteSuggestion(BaseModel):
    """Schema representing an autocomplete suggestion."""
    id: str = Field(..., description="Unique movie ID string")
    title: str = Field(..., description="Movie title")
    release_year: Optional[int] = Field(None, description="Release year of the movie")


class AutocompleteResponse(BaseModel):
    """Schema representing autocomplete responses."""
    query: str = Field(..., description="The autocomplete query string prefix")
    suggestions: List[AutocompleteSuggestion] = Field(..., description="List of matching autocomplete suggestions")


class MovieDetailResponse(BaseModel):
    """Schema representing detailed movie response containing complete metadata, poster, backdrop, and trailer URL."""
    id: str = Field(..., description="Unique movie ID string")
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

    # Additional enrichment fields
    tmdb_id: Optional[int] = Field(None, description="TMDb ID")
    imdb_id: Optional[str] = Field(None, description="IMDb ID")
    status: Optional[str] = Field(None, description="TMDb movie status")
    original_language: Optional[str] = Field(None, description="Original language")
    production_countries: List[str] = Field(default_factory=list, description="Production countries")
    budget: Optional[int] = Field(None, description="Movie budget")
    revenue: Optional[int] = Field(None, description="Movie revenue")
    homepage: Optional[str] = Field(None, description="Official homepage link")
    collection: Optional[str] = Field(None, description="Belonging collection name")
    writers: List[str] = Field(default_factory=list, description="Writers")
    producer: Optional[str] = Field(None, description="Producer name")
    composer: Optional[str] = Field(None, description="Composer name")
    cinematographer: Optional[str] = Field(None, description="Cinematographer name")
    logo_url: Optional[str] = Field(None, description="Logo URL")
    similar_movies: List[Dict[str, Any]] = Field(default_factory=list, description="Top 5 similar movies")
    recommended_movies: List[Dict[str, Any]] = Field(default_factory=list, description="Top 5 recommended movies")
    youtube_key: Optional[str] = Field(None, description="YouTube Video Key")
    trailer_type: Optional[str] = Field(None, description="Trailer type")
    trailer_name: Optional[str] = Field(None, description="Trailer name")


class MovieEnvelopeResponse(BaseModel):
    """Schema representing wrapped movie details envelope response containing execution statistics."""
    execution_statistics: dict = Field(..., description="Query execution statistics")
    movie: MovieDetailResponse = Field(..., description="Detailed movie record")


class MovieDetailCache:
    """Thread-safe LRU cache for movie details lookup."""
    def __init__(self, maxsize: int = 100) -> None:
        self.cache: OrderedDict[str, MovieEnvelopeResponse] = OrderedDict()
        self.maxsize = maxsize
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[MovieEnvelopeResponse]:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def set(self, key: str, value: MovieEnvelopeResponse) -> None:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)


movie_detail_cache = MovieDetailCache(maxsize=settings.CACHE_SIZE)


def get_movie_by_id(df: pl.DataFrame, movie_id: str) -> Optional[Dict[str, Any]]:
    """Helper to query a movie flexibly from a Polars DataFrame by TMDB ID, IMDb ID, or custom ID."""
    # 1. Match as integer tmdb_id
    try:
        tid = int(movie_id)
        filtered = df.filter(pl.col("tmdb_id") == tid)
        if filtered.height > 0:
            return filtered.to_dicts()[0]
    except ValueError:
        pass
    
    # 2. Match as string imdb_id
    filtered = df.filter(pl.col("imdb_id") == movie_id)
    if filtered.height > 0:
        return filtered.to_dicts()[0]
        
    # 3. Match as custom string/UUID id column (if exists, e.g. in tests)
    if "id" in df.columns:
        filtered = df.filter(pl.col("id") == movie_id)
        if filtered.height > 0:
            return filtered.to_dicts()[0]
            
    return None


@router.get("/movies/autocomplete", response_model=AutocompleteResponse, tags=["Movie"])
async def autocomplete_movies(
    q: str = Query(..., min_length=1, description="The prefix or title substring to autocomplete"),
    limit: int = Query(10, ge=1, le=50, description="The maximum number of autocomplete suggestions to return"),
    local_engine: LocalRetrievalEngine = Depends(get_local_retrieval_engine)
) -> AutocompleteResponse:
    """
    Returns autocomplete suggestions for movie titles matching the query substring.
    Uses local database scroll and case-insensitive matching filters.
    """
    logger.info(f"API Movie autocomplete request: q='{q}', limit={limit}")
    
    cleaned_q = q.strip()
    if not cleaned_q:
        return AutocompleteResponse(query=q, suggestions=[])
    
    if local_engine.movies_df is None:
        local_engine.initialize()

    try:
        # Case-insensitive substring match on title
        filtered = local_engine.movies_df.filter(
            pl.col("title").str.to_lowercase().str.contains(cleaned_q.lower())
        ).head(limit)
        
        suggestions = []
        for row in filtered.to_dicts():
            movie_id = str(row.get("id") or row.get("tmdb_id"))
            suggestions.append(
                AutocompleteSuggestion(
                    id=movie_id,
                    title=row.get("title", "Unknown Movie"),
                    release_year=row.get("release_year")
                )
            )
            
        return AutocompleteResponse(query=q, suggestions=suggestions)
    except Exception as e:
        logger.error(f"Error executing autocomplete: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while executing title autocomplete: {str(e)}"
        )


@router.get("/movies/{movie_id}", response_model=MovieEnvelopeResponse, tags=["Movie"])
async def get_movie_details(
    movie_id: str,
    local_engine: LocalRetrievalEngine = Depends(get_local_retrieval_engine),
    tmdb_service: TMDbService = Depends(get_tmdb_service)
) -> MovieEnvelopeResponse:
    """
    Fetches detailed metadata for a specific movie by its Qdrant UUID or TMDb ID.
    Retrieves the payload from the local database, enriches it in real-time with TMDb API,
    and returns it packaged with query execution statistics.
    """
    logger.info(f"API Movie details request for movie_id='{movie_id}'")
    start_time = time.perf_counter()

    cleaned_movie_id = movie_id.strip()
    if not cleaned_movie_id:
        raise HTTPException(status_code=400, detail="Movie ID cannot be empty or whitespace.")

    # Check cache
    cached_val = movie_detail_cache.get(cleaned_movie_id)
    if cached_val is not None:
        elapsed = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(f"[Movie Cache] HIT for movie_id: '{cleaned_movie_id}' | Time: {elapsed} ms")
        return cached_val

    if local_engine.movies_df is None:
        local_engine.initialize()

    movie_row = get_movie_by_id(local_engine.movies_df, cleaned_movie_id)
    if not movie_row:
        raise HTTPException(status_code=404, detail=f"Movie with ID '{cleaned_movie_id}' not found.")

    # Structure basic movie record dict
    movie_dict = {
        "id": str(movie_row.get("id") or movie_row.get("tmdb_id")),
        "title": movie_row.get("title", "Unknown Movie"),
        "original_title": movie_row.get("original_title"),
        "overview": movie_row.get("overview"),
        "genres": movie_row.get("genres") or [],
        "directors": movie_row.get("directors") or [],
        "cast": movie_row.get("cast") or [],
        "release_year": movie_row.get("release_year"),
        "rating_value": movie_row.get("rating_value"),
        "popularity": movie_row.get("popularity"),
        "vote_count": movie_row.get("vote_count"),
        "poster_path": movie_row.get("poster_path"),
        "tmdb_id": movie_row.get("tmdb_id"),
        "imdb_id": movie_row.get("imdb_id"),
        "runtime_minutes": movie_row.get("runtime_minutes") or movie_row.get("runtime")
    }

    # Determine source by checking cache hit
    source = "database"
    if tmdb_service.api_key and movie_dict.get("tmdb_id"):
        cached = tmdb_service.cache.get_movie_details(int(movie_dict.get("tmdb_id")))
        source = "cache" if cached is not None else "api"

    # Enrich dynamically using TMDb
    enriched_dict = await enrich_movie_with_tmdb(movie_dict, tmdb_service)
    
    elapsed_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

    response = MovieEnvelopeResponse(
        execution_statistics={
            "elapsed_time_ms": elapsed_time_ms,
            "source": source
        },
        movie=MovieDetailResponse(**enriched_dict)
    )

    movie_detail_cache.set(cleaned_movie_id, response)
    return response
