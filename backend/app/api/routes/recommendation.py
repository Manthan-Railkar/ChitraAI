import time
import math
from typing import List, Optional
from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel, Field
from loguru import logger

from app.services.recommendation_service import RecommendationService
from app.services.gemini_service import GeminiService, QueryUnderstandingResult
from app.services.tmdb_service import TMDbService
from app.vector_db.qdrant import QdrantWrapper
from app.services.enrichment_helper import enrich_movie_list
from app.api.deps import (
    get_recommendation_service,
    get_gemini_service,
    get_tmdb_service,
    get_qdrant_wrapper
)

router = APIRouter()


class RecommendationResultMovie(BaseModel):
    """Schema representing a single recommendation result with complete metadata, scores, and reason."""
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
    semantic_score: float = Field(..., description="Raw semantic cosine similarity score from Qdrant")
    boosted_semantic_score: float = Field(..., description="Semantic score with metadata boosts applied")
    reranked_score: float = Field(..., description="Final hybrid score combined with metadata weights")
    recommendation_reason: str = Field(..., description="Customized reason explaining why this movie was recommended")


class PaginationInfo(BaseModel):
    """Schema representing pagination metadata."""
    page: int = Field(default=1, description="Current page number")
    limit: int = Field(..., description="Number of results requested per page")
    total_results: int = Field(..., description="Total number of results returned")


class ExecutionStatistics(BaseModel):
    """Schema representing endpoint execution statistics."""
    elapsed_time_ms: float = Field(..., description="Elapsed execution time in milliseconds")
    source: str = Field(..., description="Data source serving the query: 'cache', 'api', or 'database'")


class RecommendationResponse(BaseModel):
    """Schema representing the complete semantic recommendation response."""
    query: str = Field(..., description="The original natural-language query request or movie ID")
    pagination: PaginationInfo = Field(..., description="Pagination metadata")
    execution_statistics: ExecutionStatistics = Field(..., description="Query execution statistics")
    understanding: Optional[QueryUnderstandingResult] = Field(None, description="Structured parameters parsed by Gemini (only for semantic requests)")
    results: List[RecommendationResultMovie] = Field(..., description="Ranked and filtered movie recommendations")


@router.get("/recommendations/semantic", response_model=RecommendationResponse, tags=["Recommendation"])
async def get_semantic_recommendations(
    q: str = Query(..., min_length=1, description="The natural language query describing movie preferences"),
    limit: int = Query(10, ge=1, le=100, description="The maximum number of recommended movies to return"),
    gemini_service: GeminiService = Depends(get_gemini_service),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    tmdb_service: TMDbService = Depends(get_tmdb_service)
) -> RecommendationResponse:
    """
    Computes structured recommendations from a natural language request. 
    Maintains clean separation by first calling the Gemini Query Understanding module to build 
    the parameters, and then invoking the recommendation engine to build the query, retrieve candidates, 
    filter, boost, rerank, and explain matches. Enriches final results dynamically using TMDb.
    """
    logger.info(f"FastAPI semantic recommendation endpoint hit: q='{q}', limit={limit}")
    start_time = time.perf_counter()
    
    try:
        # Step 1: Query Understanding (Separation of concern)
        understanding = await gemini_service.understand_query(query=q)
        
        # Step 2: Retrieve candidate recommendations from parameters
        results = await recommendation_service.recommend_movies_from_understanding(
            understanding=understanding,
            limit=limit
        )

        # Determine source by checking cache hits
        source = "database"
        if results and tmdb_service.api_key:
            source = "cache"
            for movie in results:
                tmdb_id = movie.get("tmdb_id")
                if tmdb_id:
                    cached = tmdb_service.cache.get_movie_details(int(tmdb_id))
                    if cached is None:
                        source = "api"
                        break

        # Step 3: Dynamic real-time TMDb enrichment
        enriched_results = await enrich_movie_list(results, tmdb_service)
        
        # Step 4: Map to Response Models
        formatted_results = [RecommendationResultMovie(**movie) for movie in enriched_results]
        
        elapsed_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return RecommendationResponse(
            query=q,
            pagination=PaginationInfo(
                page=1,
                limit=limit,
                total_results=len(formatted_results)
            ),
            execution_statistics=ExecutionStatistics(
                elapsed_time_ms=elapsed_time_ms,
                source=source
            ),
            understanding=understanding,
            results=formatted_results
        )
    except Exception as e:
        logger.error(f"Unexpected error in FastAPI semantic recommendation endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while generating recommendation results: {str(e)}"
        )


@router.get("/recommendations/movie/{movie_id}", response_model=RecommendationResponse, tags=["Recommendation"])
async def get_recommendations_by_movie(
    movie_id: str,
    limit: int = Query(10, ge=1, le=100, description="The maximum number of recommended movies to return"),
    qdrant_service: QdrantWrapper = Depends(get_qdrant_wrapper),
    tmdb_service: TMDbService = Depends(get_tmdb_service)
) -> RecommendationResponse:
    """
    Recommends movies similar to a given source movie point UUID.
    Retrieves the target movie from Qdrant, runs nearest-neighbor recommendation query,
    computes soft boosting based on overlapping attributes with the source movie,
    applies hybrid metadata reranking, and enriches details via TMDb in real-time.
    """
    logger.info(f"API similar movies request for movie_id='{movie_id}', limit={limit}")
    start_time = time.perf_counter()

    # Connect to Qdrant if needed
    if not qdrant_service.client:
        qdrant_service.connect()

    # 1. Retrieve the source movie to compare metadata
    try:
        points = qdrant_service.client.retrieve(
            collection_name=qdrant_service.collection_name,
            ids=[movie_id]
        )
    except Exception as e:
        logger.error(f"Error retrieving source movie point: {e}")
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")

    if not points:
        raise HTTPException(status_code=404, detail=f"Source movie with ID '{movie_id}' not found.")

    source_movie = points[0]
    source_payload = source_movie.payload or {}
    source_title = source_payload.get("title", "the original movie")
    source_genres = [g.lower() for g in source_payload.get("genres", [])]
    source_cast = [c.lower() for c in source_payload.get("cast", [])]
    source_directors = [d.lower() for d in source_payload.get("directors", [])]

    # 2. Retrieve similar points in Qdrant (using query_points with ID is recommended)
    try:
        # Get point-similar candidates
        response = qdrant_service.client.query_points(
            collection_name=qdrant_service.collection_name,
            query=movie_id,
            limit=max(50, limit * 5)
        )
        scored_points = response.points
    except Exception as e:
        logger.error(f"Error executing Qdrant similar points search: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation query error: {str(e)}")

    # 3. Apply Soft Boosting, Hybrid Reranking, and reasons
    results = []
    for hit in scored_points:
        payload = hit.payload or {}
        
        # Avoid recommending the source movie itself
        if str(hit.id) == movie_id:
            continue

        base_semantic_score = hit.score
        boost = 0.0
        
        matched_genres = []
        matched_actors = []
        matched_directors = []

        # Genres overlap
        for g in payload.get("genres", []):
            if g.lower() in source_genres:
                boost += 0.03
                matched_genres.append(g)

        # Cast overlap
        for c in payload.get("cast", []):
            if c.lower() in source_cast:
                boost += 0.05
                matched_actors.append(c)

        # Directors overlap
        for d in payload.get("directors", []):
            if d.lower() in source_directors:
                boost += 0.05
                matched_directors.append(d)

        boosted_semantic_score = min(1.0, base_semantic_score + boost)

        # Hybrid Reranking
        rating_value = payload.get("rating_value")
        popularity = payload.get("popularity")
        vote_count = payload.get("vote_count")

        rating_val = float(rating_value) if rating_value is not None else 5.0
        pop_val = float(popularity) if popularity is not None else 0.0
        votes_val = int(vote_count) if vote_count is not None else 0

        score_rating = rating_val / 10.0
        score_popularity = min(1.0, math.log1p(pop_val) / 5.0)
        score_votes = min(1.0, math.log1p(votes_val) / 15.0)

        reranked_score = (
            0.6 * boosted_semantic_score +
            0.2 * score_rating +
            0.1 * score_popularity +
            0.1 * score_votes
        )

        # Construct customized reason string
        overlaps = []
        if matched_genres:
            overlaps.append(f"genres ({', '.join(matched_genres)})")
        if matched_actors:
            overlaps.append(f"cast ({', '.join(matched_actors)})")
        if matched_directors:
            overlaps.append(f"directors ({', '.join(matched_directors)})")

        if overlaps:
            reason = f"Recommended because it shares {', and '.join(overlaps)} with {source_title}."
        else:
            reason = f"Recommended due to strong thematic similarities to {source_title}."

        movie_dict = {
            "id": str(hit.id),
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
            "semantic_score": base_semantic_score,
            "boosted_semantic_score": round(boosted_semantic_score, 4),
            "reranked_score": round(reranked_score, 4),
            "recommendation_reason": reason
        }
        results.append(movie_dict)

    # Sort results
    results.sort(key=lambda x: x["reranked_score"], reverse=True)
    sliced_results = results[:limit]

    # 4. Enrich results
    source = "database"
    if sliced_results and tmdb_service.api_key:
        source = "cache"
        for movie in sliced_results:
            tmdb_id = movie.get("tmdb_id")
            if tmdb_id:
                cached = tmdb_service.cache.get_movie_details(int(tmdb_id))
                if cached is None:
                    source = "api"
                    break

    enriched_results = await enrich_movie_list(sliced_results, tmdb_service)
    formatted_results = [RecommendationResultMovie(**movie) for movie in enriched_results]

    elapsed_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return RecommendationResponse(
        query=f"movie:{movie_id}",
        pagination=PaginationInfo(
            page=1,
            limit=limit,
            total_results=len(formatted_results)
        ),
        execution_statistics=ExecutionStatistics(
            elapsed_time_ms=elapsed_time_ms,
            source=source
        ),
        results=formatted_results
    )


@router.get("/recommendations/user/{user_id}", tags=["Recommendation"])
async def get_recommendations_by_user(
    user_id: str,
    limit: int = Query(10, description="The maximum number of recommended movies to return")
):
    """
    Legacy mock placeholder. Recommends movies based on the user's historical preferences.
    """
    logger.info(f"API Recommendation request for user_id='{user_id}', limit={limit}")
    return {
        "user_id": user_id,
        "results": [
            {
                "id": "mock-4",
                "title": "The Matrix",
                "overview": "A computer hacker learns from mysterious rebels about the true nature of his reality.",
                "score": 0.91
            }
        ]
    }
