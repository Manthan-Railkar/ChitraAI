import time
import math
import uuid
import threading
import polars as pl
import numpy as np
from collections import OrderedDict
from typing import List, Optional, Any, Dict, Tuple
from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel, Field
from loguru import logger

from app.core.config import settings
from app.services.recommendation_service import RecommendationService
from app.services.tmdb_service import TMDbService
from app.services.openai_service import OpenAIService
from app.services.local_retrieval import LocalRetrievalEngine, SemanticSimilarityCalculator, build_embedding_document
from app.services.enrichment_helper import enrich_movie_list
from app.api.deps import (
    get_recommendation_service,
    get_tmdb_service,
    get_local_retrieval_engine,
    get_openai_service
)
from app.api.routes.query import QueryUnderstandingResult, YearConstraints

router = APIRouter()


class StandardizedMovie(BaseModel):
    """Schema representing a single standardized movie recommendation result."""
    tmdb_id: int = Field(..., description="TMDb movie ID")
    title: str = Field(..., description="Movie title")
    overview: Optional[str] = Field(None, description="Synopsis or overview")
    poster_path: Optional[str] = Field(None, description="TMDb poster suffix path")
    genres: List[str] = Field(default_factory=list, description="Movie genres")
    runtime: Optional[int] = Field(None, description="Runtime in minutes")
    release_year: Optional[int] = Field(None, description="Release year")
    rating: Optional[float] = Field(None, description="Average rating in range [1.0, 10.0]")
    popularity: Optional[float] = Field(None, description="TMDb popularity score")
    retrieval_score: float = Field(..., description="Reranked semantic score")
    confidence_score: float = Field(..., description="Confidence similarity score")
    recommendation_reason: Optional[str] = Field(None, description="Explanation for this recommendation")

    # Additional existing fields (preserved to avoid losing metadata or breaking tests)
    id: Optional[str] = Field(None, description="UUID string or tmdb ID string")
    original_title: Optional[str] = Field(None, description="Original movie title")
    directors: List[str] = Field(default_factory=list, description="List of directors")
    cast: List[str] = Field(default_factory=list, description="List of cast members")
    vote_count: Optional[int] = Field(None, description="Total vote count")
    backdrop_path: Optional[str] = Field(None, description="Backdrop image suffix path")
    trailer_url: Optional[str] = Field(None, description="YouTube trailer link")
    streaming_providers: List[str] = Field(default_factory=list, description="Streaming providers in the US")
    certification: Optional[str] = Field(None, description="US age certification")


class RecommendationResponse(BaseModel):
    """Schema representing the standardized semantic recommendation response."""
    success: bool = Field(..., description="Indicates request success status")
    message: str = Field(..., description="Descriptive status message")
    query: str = Field(..., description="The original natural-language query request or movie ID")
    recommendations: List[StandardizedMovie] = Field(..., description="Ranked and filtered movie recommendations")
    metadata: dict = Field(..., description="Detailed execution statistics, pagination, and intent info")


class RecommendationRequest(BaseModel):
    """Schema representing standard recommendations POST request payload."""
    query: str = Field(..., min_length=1, description="The natural language query describing movie preferences")
    limit: int = Field(10, ge=1, le=100, description="The maximum number of recommended movies to return")


class RecommendationRequestCache:
    """Thread-safe LRU cache for final recommendation responses."""
    def __init__(self, maxsize: int = 100) -> None:
        self.cache: OrderedDict[str, RecommendationResponse] = OrderedDict()
        self.maxsize = maxsize
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[RecommendationResponse]:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def set(self, key: str, value: RecommendationResponse) -> None:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)


recommendation_cache = RecommendationRequestCache(maxsize=100)


def map_to_standardized_movie(movie: dict) -> StandardizedMovie:
    """Maps enriched movie details dictionary to standard model fields."""
    return StandardizedMovie(
        tmdb_id=int(movie.get("tmdb_id") or 0),
        title=movie.get("title", "Unknown Movie"),
        overview=movie.get("overview"),
        poster_path=movie.get("poster_path"),
        genres=movie.get("genres") or [],
        runtime=movie.get("runtime_minutes") or movie.get("runtime"),
        release_year=movie.get("release_year"),
        rating=movie.get("rating_value") or movie.get("rating"),
        popularity=movie.get("popularity"),
        retrieval_score=movie.get("reranked_score") or movie.get("retrieval_score") or 0.0,
        confidence_score=movie.get("confidence_score") or movie.get("boosted_semantic_score") or 0.0,
        recommendation_reason=movie.get("recommendation_reason"),
        
        # Preserve legacy details fields
        id=movie.get("id"),
        original_title=movie.get("original_title"),
        directors=movie.get("directors") or [],
        cast=movie.get("cast") or [],
        vote_count=movie.get("vote_count"),
        backdrop_path=movie.get("backdrop_path"),
        trailer_url=movie.get("trailer_url"),
        streaming_providers=movie.get("streaming_providers") or [],
        certification=movie.get("certification")
    )


def evaluate_semantic_confidence(results: List[Dict[str, Any]], limit: int) -> float:
    """
    Evaluates the confidence score (0.0 to 1.0) of semantic search results
    based on similarity scores, candidate count, and metadata completeness.
    """
    if not results:
        return 0.0

    top_movie = results[0]
    top_score = top_movie.get("semantic_score", 0.0)
    count_ratio = min(len(results) / limit, 1.0)
    
    completeness_scores = []
    for movie in results[:3]:
        fields = ["title", "overview", "genres"]
        valid_fields = sum(1 for f in fields if movie.get(f))
        completeness_scores.append(valid_fields / len(fields))
    avg_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 1.0

    confidence = (top_score * 0.7) + (count_ratio * 0.15) + (avg_completeness * 0.15)
    return round(confidence, 4)


import json
from pathlib import Path

def log_structured_debug_report(report: dict) -> None:
    """Writes a structured log of the recommendation request to logs/recommendation_debug.log."""
    try:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "recommendation_debug.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Failed to write structured debug log: {e}")

async def execute_recommendation_flow(
    query: str,
    limit: int,
    recommendation_service: RecommendationService,
    tmdb_service: TMDbService
) -> RecommendationResponse:
    """Core recommendation execution pipeline shared across GET and POST endpoints."""
    total_start = time.perf_counter()
    
    # Check request cache first
    cache_key = f"{query}_{limit}"
    cached_resp = recommendation_cache.get(cache_key)
    if cached_resp is not None:
        elapsed = round((time.perf_counter() - total_start) * 1000, 2)
        logger.info(f"[TIMING] Served from Request Cache: {elapsed} ms")
        return cached_resp

    try:
        # Step 1 & 2: Parse intent & retrieve candidates from local retrieval engine
        intent, results = await recommendation_service.recommend_movies_from_query(query=query, limit=limit)

        # Step 3: Map intent to QueryUnderstandingResult for API envelope compatibility
        search_intent = getattr(intent, "intent", "recommendation")
        if not search_intent or search_intent == "unknown":
            has_params = bool(
                intent.genres or intent.moods or intent.themes or
                intent.preferred_actors or intent.preferred_directors or
                intent.similar_movies
            )
            search_intent = "recommendation" if has_params else "search"

        understanding = QueryUnderstandingResult(
            search_intent=search_intent,
            mood=", ".join(intent.moods) if intent.moods else None,
            genres=intent.genres,
            excluded_genres=intent.avoid_genres,
            themes=intent.themes,
            actors=intent.preferred_actors,
            directors=intent.preferred_directors,
            reference_movies=intent.similar_movies,
            user_preferences=f"Runtime: {intent.runtime} min" if intent.runtime else None,
            intent=search_intent,
            ranking_mode=getattr(intent, "ranking_mode", "default"),
            release_year=getattr(intent, "release_year", None),
            exclusions=getattr(intent, "exclusions", [])
        )
        if intent.year_range:
            understanding.release_year_constraints = YearConstraints(
                start_year=intent.year_range.start,
                end_year=intent.year_range.end
            )

        # Step 4: Determine source (database, cache, or api)
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

        # Step 5: Dynamic real-time TMDb enrichment
        t_enrich_start = time.perf_counter()
        enriched_results = await enrich_movie_list(results, tmdb_service)
        time_enrich = round((time.perf_counter() - t_enrich_start) * 1000, 2)
        logger.info(f"[TIMING] TMDb Enrichment: {time_enrich} ms")
        
        # Step 6: Map to Response Models
        formatted_results = [map_to_standardized_movie(movie) for movie in enriched_results]
        confidence_score = evaluate_semantic_confidence(results, limit)
        elapsed_time_ms = round((time.perf_counter() - total_start) * 1000, 2)

        # Retrieve/build debug report
        engine = recommendation_service.local_retrieval_engine
        debug_report = getattr(engine, "last_debug_report", None)
        
        # If it was movie_lookup bypass, construct a direct search debug report
        if search_intent == "movie_lookup" or not debug_report or debug_report.get("query") != query:
            debug_report = {
                "query": query,
                "intent": intent.model_dump() if hasattr(intent, "model_dump") else str(intent),
                "message": "Movie lookup direct search executed.",
                "execution_time_ms": elapsed_time_ms
            }
            if engine:
                engine.last_debug_report = debug_report

        if debug_report:
            log_structured_debug_report(debug_report)

        metadata = {
            "pagination": {
                "page": 1,
                "limit": limit,
                "total_results": len(formatted_results)
            },
            "execution_statistics": {
                "elapsed_time_ms": elapsed_time_ms,
                "source": source
            },
            "confidence_score": confidence_score,
            "understanding": understanding.model_dump() if understanding else None
        }

        if debug_report and settings.LOG_LEVEL.upper() == "DEBUG":
            metadata["debug_report"] = debug_report

        response = RecommendationResponse(
            success=True,
            message="Recommendations retrieved successfully.",
            query=query,
            recommendations=formatted_results,
            metadata=metadata
        )
        recommendation_cache.set(cache_key, response)
        return response

    except Exception as e:
        logger.exception("Unexpected error in recommendation pipeline")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while generating recommendation results: {str(e)}"
        )


@router.get("/recommendations/semantic", response_model=RecommendationResponse, tags=["Recommendation"])
async def get_semantic_recommendations(
    q: str = Query(..., min_length=1, description="The natural language query describing movie preferences"),
    limit: int = Query(10, ge=1, le=100, description="The maximum number of recommended movies to return"),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    tmdb_service: TMDbService = Depends(get_tmdb_service)
) -> RecommendationResponse:
    """
    Computes structured recommendations from a natural language request using the local retrieval engine.
    """
    logger.info(f"FastAPI semantic recommendation GET endpoint hit: q='{q}', limit={limit}")
    return await execute_recommendation_flow(q, limit, recommendation_service, tmdb_service)


@router.post("/recommendations", response_model=RecommendationResponse, tags=["Recommendation"])
async def post_recommendations(
    req: RecommendationRequest,
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    tmdb_service: TMDbService = Depends(get_tmdb_service)
) -> RecommendationResponse:
    """
    Computes structured recommendations from a POST request payload.
    """
    logger.info(f"FastAPI semantic recommendation POST endpoint hit: query='{req.query}', limit={req.limit}")
    return await execute_recommendation_flow(req.query, req.limit, recommendation_service, tmdb_service)


@router.post("/recommend", response_model=RecommendationResponse, tags=["Recommendation"])
async def post_recommend(
    req: RecommendationRequest,
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    tmdb_service: TMDbService = Depends(get_tmdb_service)
) -> RecommendationResponse:
    """
    Alias endpoint for POST /recommendations.
    """
    return await execute_recommendation_flow(req.query, req.limit, recommendation_service, tmdb_service)


@router.get("/recommendations/movie/{movie_id}", response_model=RecommendationResponse, tags=["Recommendation"])
async def get_recommendations_by_movie(
    movie_id: str,
    limit: int = Query(10, ge=1, le=100, description="The maximum number of recommended movies to return"),
    local_engine: LocalRetrievalEngine = Depends(get_local_retrieval_engine),
    tmdb_service: TMDbService = Depends(get_tmdb_service)
) -> RecommendationResponse:
    """
    Recommends movies similar to a given source movie ID.
    Retrieves the target movie from the local database, runs nearest-neighbor recommendation query,
    computes soft boosting based on overlapping attributes with the source movie,
    applies hybrid metadata reranking, and enriches details via TMDb in real-time.
    """
    logger.info(f"API similar movies request for movie_id='{movie_id}', limit={limit}")
    start_time = time.perf_counter()

    if local_engine.movies_df is None:
        local_engine.initialize()

    # 1. Retrieve the source movie to compare metadata
    from app.api.routes.movie import get_movie_by_id
    source_movie = get_movie_by_id(local_engine.movies_df, movie_id)
    if not source_movie:
        raise HTTPException(status_code=404, detail=f"Source movie with ID '{movie_id}' not found.")

    source_title = source_movie.get("title", "the original movie")
    source_genres = [g.lower() for g in (source_movie.get("genres") or [])]
    source_cast = [c.lower() for c in (source_movie.get("cast") or [])]
    source_directors = [d.lower() for d in (source_movie.get("directors") or [])]
    source_tmdb_id = source_movie.get("tmdb_id")

    # 2. Extract vector embedding of the source movie
    source_vector = None
    if source_tmdb_id in local_engine.tmdb_id_to_idx:
        idx = local_engine.tmdb_id_to_idx[source_tmdb_id]
        source_vector = local_engine.embeddings_matrix[idx]
    else:
        doc = build_embedding_document(
            title=source_movie.get("title"),
            tagline=source_movie.get("tagline"),
            overview=source_movie.get("overview"),
            genres=source_movie.get("genres") or [],
            keywords=source_movie.get("keywords") or [],
            cast=source_movie.get("cast") or [],
            directors=source_movie.get("directors") or []
        )
        source_vector = local_engine.embedding_service.encode_single(doc, normalize=True)

    # 3. Retrieve similar points in local engine (excluding the source movie)
    candidates_df = local_engine.movies_df.filter(pl.col("tmdb_id") != source_tmdb_id)
    valid_tmdb_ids = list(local_engine.tmdb_id_to_idx.keys())
    candidates_df = candidates_df.filter(pl.col("tmdb_id").is_in(valid_tmdb_ids))
    
    if candidates_df.height == 0:
        return RecommendationResponse(
            success=True,
            message="No recommendations found.",
            query=f"movie:{movie_id}",
            recommendations=[],
            metadata={
                "pagination": {"page": 1, "limit": limit, "total_results": 0},
                "execution_statistics": {
                    "elapsed_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
                    "source": "database"
                }
            }
        )

    candidate_ids = candidates_df.select("tmdb_id").to_series().to_list()
    candidate_indices = [local_engine.tmdb_id_to_idx[tid] for tid in candidate_ids]
    candidate_embs = local_engine.embeddings_matrix[candidate_indices]

    semantic_scores = SemanticSimilarityCalculator.compute_similarities(source_vector, candidate_embs)
    candidates_df = candidates_df.with_columns(pl.Series("semantic_score", semantic_scores))
    if candidates_df.height > 1000:
        candidates_df = candidates_df.sort("semantic_score", descending=True).head(1000)
    
    # 4. Apply Soft Boosting, Hybrid Reranking, and reasons
    results = []
    for row in candidates_df.to_dicts():
        base_semantic_score = float(row["semantic_score"])
        boost = 0.0
        
        matched_genres = []
        matched_actors = []
        matched_directors = []

        # Genres overlap
        for g in (row.get("genres") or []):
            if g.lower() in source_genres:
                boost += 0.03
                matched_genres.append(g)

        # Cast overlap
        for c in (row.get("cast") or []):
            if c.lower() in source_cast:
                boost += 0.05
                matched_actors.append(c)

        # Directors overlap
        for d in (row.get("directors") or []):
            if d.lower() in source_directors:
                boost += 0.05
                matched_directors.append(d)

        boosted_semantic_score = min(1.0, base_semantic_score + boost)

        # Hybrid Reranking
        rating_value = row.get("rating_value")
        popularity = row.get("popularity")
        vote_count = row.get("vote_count")

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
            "id": str(row.get("id") or row.get("tmdb_id")),
            "title": row.get("title", "Unknown Movie"),
            "original_title": row.get("original_title"),
            "overview": row.get("overview"),
            "genres": row.get("genres") or [],
            "directors": row.get("directors") or [],
            "cast": row.get("cast") or [],
            "release_year": row.get("release_year"),
            "rating_value": rating_value,
            "popularity": popularity,
            "vote_count": vote_count,
            "poster_path": row.get("poster_path"),
            "tmdb_id": row.get("tmdb_id"),
            "semantic_score": base_semantic_score,
            "boosted_semantic_score": round(boosted_semantic_score, 4),
            "reranked_score": round(reranked_score, 4),
            "recommendation_reason": reason
        }
        results.append(movie_dict)

    # Sort results
    results.sort(key=lambda x: x["reranked_score"], reverse=True)
    sliced_results = results[:limit]

    # 5. Enrich results
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
    formatted_results = [map_to_standardized_movie(movie) for movie in enriched_results]

    elapsed_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # Format and save debug report
    def format_debug_candidate(movie: dict, final_score: float = 0.0) -> dict:
        return {
            "title": movie.get("title"),
            "tmdb_id": movie.get("tmdb_id"),
            "genres": movie.get("genres") or [],
            "semantic_score": round(float(movie.get("semantic_score") or 0.0), 4),
            "bm25_score": 0.0,
            "metadata_match_score": 0.0,
            "rating": movie.get("rating_value") or movie.get("rating"),
            "vote_count": movie.get("vote_count"),
            "popularity": movie.get("popularity"),
            "final_score": round(float(final_score), 4)
        }

    top_sem_candidates = sorted(results, key=lambda x: x["semantic_score"], reverse=True)[:100]
    
    debug_report = {
        "query": f"movie:{movie_id}",
        "intent": {
            "intent": "similar_movie",
            "similar_movies": [source_title]
        },
        "top_100_metadata_candidates": [],
        "top_100_semantic_candidates": [format_debug_candidate(m) for m in top_sem_candidates],
        "top_100_bm25_candidates": [],
        "candidate_fusion_output": [format_debug_candidate(m, final_score=m.get("reranked_score") or 0.0) for m in results[:100]],
        "top_20_ranked_movies": [format_debug_candidate(m, final_score=m.get("reranked_score") or 0.0) for m in results[:20]],
        "selected_top_3": [format_debug_candidate(m, final_score=m.get("reranked_score") or 0.0) for m in sliced_results[:3]],
        "execution_time_ms": elapsed_time_ms
    }

    if local_engine:
        local_engine.last_debug_report = debug_report

    log_structured_debug_report(debug_report)

    metadata = {
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

    if settings.LOG_LEVEL.upper() == "DEBUG":
        metadata["debug_report"] = debug_report

    return RecommendationResponse(
        success=True,
        message="Recommendations retrieved successfully.",
        query=f"movie:{movie_id}",
        recommendations=formatted_results,
        metadata=metadata
    )


@router.get("/recommendations/user/{user_id}", response_model=RecommendationResponse, tags=["Recommendation"])
async def get_recommendations_by_user(
    user_id: str,
    limit: int = Query(10, description="The maximum number of recommended movies to return")
) -> RecommendationResponse:
    """
    Legacy mock placeholder. Recommends movies based on the user's historical preferences.
    """
    logger.info(f"API Recommendation request for user_id='{user_id}', limit={limit}")
    return RecommendationResponse(
        success=True,
        message="User recommendations retrieved successfully (mock mode).",
        query=f"user:{user_id}",
        recommendations=[
            StandardizedMovie(
                tmdb_id=603,
                title="The Matrix",
                overview="A computer hacker learns from mysterious rebels about the true nature of his reality.",
                poster_path="/mock-poster.jpg",
                genres=["Action", "Sci-Fi"],
                runtime=136,
                release_year=1999,
                rating=8.7,
                popularity=80.0,
                retrieval_score=0.91,
                confidence_score=0.91,
                recommendation_reason="Mock recommendation."
            )
        ],
        metadata={
            "pagination": {"page": 1, "limit": limit, "total_results": 1},
            "execution_statistics": {"elapsed_time_ms": 1.0, "source": "database"}
        }
    )
