from typing import Any, List, Dict, Optional, Tuple
from loguru import logger
import polars as pl

from app.services.local_retrieval import LocalRetrievalEngine
from app.services.intent_extractor import IntentExtractor, RecommendationIntent
from app.services.tmdb_service import TMDbService
from app.services.enrichment_helper import enrich_movie_with_tmdb


class RecommendationService:
    """
    Local recommendation engine for ChitraAI.
    Combines OpenAI intent extraction, structured filtering, local semantic similarity,
    and weighted scoring to retrieve the best candidate recommendations.
    """
    def __init__(self, local_retrieval_engine: LocalRetrievalEngine, intent_extractor: IntentExtractor, tmdb_service: Optional[TMDbService] = None) -> None:
        self.local_retrieval_engine = local_retrieval_engine
        self.intent_extractor = intent_extractor
        self.tmdb_service = tmdb_service or TMDbService()
        logger.info("RecommendationService initialized with local retrieval engine and OpenAI intent extractor.")

    async def recommend_movies_from_query(self, query: str, limit: int = 50) -> Tuple[RecommendationIntent, List[Dict[str, Any]]]:
        """
        Extracts recommendation intent using OpenAI, runs structured filtering + semantic search,
        and returns the extracted intent along with the top candidate recommendations.
        """
        logger.info(f"Generating recommendations for query: '{query}' (limit={limit})")
        
        # 1. Extract structured intent using OpenAI
        intent = await self.intent_extractor.extract_intent(query)
        
        # 1b. Check if the intent is movie_lookup
        if getattr(intent, "intent", "recommendation") == "movie_lookup":
            search_title = intent.similar_movies[0] if intent.similar_movies else query
            search_year = getattr(intent, "release_year", None)
            if not search_year and intent.year_range:
                search_year = intent.year_range.start or intent.year_range.end
                
            logger.info(f"[Movie Lookup Bypass] Query '{query}' classified as movie_lookup. Searching TMDb for title='{search_title}', year={search_year}")
            
            tmdb_id = None
            movie_details = None
            if self.tmdb_service:
                try:
                    tmdb_id = await self.tmdb_service.search_movie_by_title(search_title, year=search_year)
                    if tmdb_id:
                        movie_details = await self.tmdb_service.fetch_movie_details(tmdb_id)
                except Exception as e:
                    logger.error(f"[Movie Lookup Bypass] Failed to search or fetch from TMDb: {e}")
            
            if movie_details:
                movie_dict = {
                    "tmdb_id": tmdb_id,
                    "title": movie_details.get("title", search_title),
                    "original_title": movie_details.get("original_title"),
                    "overview": movie_details.get("overview"),
                    "genres": [g.get("name") for g in movie_details.get("genres", []) if g.get("name")],
                    "release_year": int(movie_details.get("release_date", "0000")[:4]) if movie_details.get("release_date") else None,
                    "rating_value": movie_details.get("vote_average"),
                    "vote_count": movie_details.get("vote_count"),
                    "popularity": movie_details.get("popularity"),
                    "poster_path": movie_details.get("poster_path"),
                    "backdrop_path": movie_details.get("backdrop_path"),
                    "runtime_minutes": movie_details.get("runtime"),
                    "recommendation_reason": "Exact match found in TMDb lookup.",
                    "retrieval_score": 1.0,
                    "boosted_semantic_score": 1.0,
                    "reranked_score": 1.0
                }
                
                credits = movie_details.get("credits", {})
                if credits:
                    directors = [m.get("name") for m in credits.get("crew", []) if m.get("job") == "Director"]
                    movie_dict["directors"] = directors
                    cast = [m.get("name") for m in credits.get("cast", [])[:10]]
                    movie_dict["cast"] = cast
                
                enriched_movie = await enrich_movie_with_tmdb(movie_dict, self.tmdb_service)
                return intent, [enriched_movie]
            
            # Local Database Fallback
            logger.info(f"[Movie Lookup Fallback] Movie '{search_title}' not found via TMDb search. Falling back to local database search...")
            if self.local_retrieval_engine.movies_df is None:
                self.local_retrieval_engine.initialize()
            
            matched_df = self.local_retrieval_engine.movies_df.filter(
                pl.col("title").str.to_lowercase().str.contains(search_title.lower())
            )
            if matched_df.height > 0:
                best_match = matched_df.sort("popularity", descending=True).to_dicts()[0]
                best_match["recommendation_reason"] = "Exact match found in local database lookup."
                best_match["retrieval_score"] = 1.0
                best_match["boosted_semantic_score"] = 1.0
                best_match["reranked_score"] = 1.0
                
                enriched_movie = await enrich_movie_with_tmdb(best_match, self.tmdb_service)
                return intent, [enriched_movie]
                
            logger.warning(f"[Movie Lookup Fallback] Movie '{search_title}' not found in TMDb or local database.")
            return intent, []

        # 2. Retrieve candidates from TMDb Discover if enabled, otherwise fallback to local engine
        from app.core.config import settings
        if settings.USE_TMDB_RETRIEVAL and self.tmdb_service and self.tmdb_service.api_key:
            try:
                logger.info("[TMDb Retrieval] Using TMDb Discover API for candidate retrieval.")
                from app.services.tmdb_query_builder import TMDbQueryBuilder
                discover_params = await TMDbQueryBuilder.build_query(intent, self.tmdb_service)
                logger.info(f"[TMDb Retrieval] Discover params: {discover_params}")

                discover_response = await self.tmdb_service.discover_movies(discover_params)
                
                candidates = []
                if discover_response:
                    results_list = discover_response.get("results", [])
                    target_candidates = results_list[:30]
                    
                    import asyncio
                    tasks = [self.tmdb_service.fetch_movie_details(m.get("id")) for m in target_candidates if m.get("id")]
                    details_list = await asyncio.gather(*tasks)
                    
                    for details in details_list:
                        if not details:
                            continue
                        tmdb_id = details.get("id")
                        movie_dict = {
                            "tmdb_id": tmdb_id,
                            "title": details.get("title"),
                            "original_title": details.get("original_title"),
                            "overview": details.get("overview"),
                            "genres": [g.get("name") for g in details.get("genres", []) if g.get("name")],
                            "release_year": int(details.get("release_date", "0000")[:4]) if details.get("release_date") else None,
                            "rating_value": details.get("vote_average"),
                            "vote_count": details.get("vote_count"),
                            "popularity": details.get("popularity"),
                            "poster_path": details.get("poster_path"),
                            "backdrop_path": details.get("backdrop_path"),
                            "tagline": details.get("tagline"),
                            "keywords": [k.get("name") for k in details.get("keywords", {}).get("keywords", []) if k.get("name")],
                            "cast": [m.get("name") for m in details.get("credits", {}).get("cast", [])[:10]],
                            "directors": [m.get("name") for m in details.get("credits", {}).get("crew", []) if m.get("job") == "Director"]
                        }
                        candidates.append(movie_dict)

                if candidates:
                    query_doc = f"{query}. Genres: {', '.join(intent.genres)}. Themes: {', '.join(intent.themes)}. Moods: {', '.join(intent.moods)}. Keywords: {', '.join(intent.keywords)}."
                    query_vector = self.local_retrieval_engine.embedding_service.encode_single(query_doc, normalize=True)
                    
                    def build_temp_document(movie: dict) -> str:
                        parts = [f"Title: {movie.get('title')}"]
                        if movie.get("genres"):
                            parts.append(f"Genres: {', '.join(movie.get('genres'))}")
                        if movie.get("directors"):
                            parts.append(f"Directed by: {', '.join(movie.get('directors'))}")
                        if movie.get("cast"):
                            parts.append(f"Starring: {', '.join(movie.get('cast'))}")
                        if movie.get("keywords"):
                            parts.append(f"Keywords: {', '.join(movie.get('keywords'))}")
                        if movie.get("overview"):
                            parts.append(movie.get("overview"))
                        return "\n".join(parts)
                    
                    candidate_docs = [build_temp_document(m) for m in candidates]
                    candidate_embs = self.local_retrieval_engine.embedding_service.encode_batch(candidate_docs, normalize=True)
                    
                    import numpy as np
                    semantic_scores = np.dot(candidate_embs, query_vector)
                    
                    from app.services.ranking_service import RankingService
                    scored_candidates = RankingService.rank_candidates(candidates, intent, semantic_scores)
                    
                    diversified_candidates = RankingService.apply_diversity(scored_candidates)
                    top_candidates = diversified_candidates[:limit]
                    
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

                    self.local_retrieval_engine.last_debug_report = {
                        "query": query,
                        "intent": intent.model_dump() if hasattr(intent, "model_dump") else str(intent),
                        "top_100_metadata_candidates": [],
                        "top_100_semantic_candidates": [format_debug_candidate(m) for m in candidates],
                        "top_100_bm25_candidates": [],
                        "candidate_fusion_output": [format_debug_candidate(m, final_score=m.get("retrieval_score") or 0.0) for m in scored_candidates],
                        "top_20_ranked_movies": [format_debug_candidate(m, final_score=m.get("retrieval_score") or 0.0) for m in scored_candidates[:20]],
                        "selected_top_3": [format_debug_candidate(m, final_score=m.get("retrieval_score") or 0.0) for m in top_candidates],
                        "execution_time_ms": 0.0
                    }
                    
                    logger.info(f"[TMDb Retrieval] Success. Candidates retrieved: {len(top_candidates)}")
                    from app.services.enrichment_helper import enrich_movie_list
                    enriched_candidates = await enrich_movie_list(top_candidates, self.tmdb_service)
                    return intent, enriched_candidates
                else:
                    logger.warning("[TMDb Retrieval] Discover returned 0 candidates. Falling back to local retrieval...")
            except Exception as e:
                logger.error(f"[TMDb Retrieval] Discover retrieval failed: {e}. Falling back to local retrieval...")

        # 3. Retrieve candidates locally (Fallback)
        results = await self.local_retrieval_engine.retrieve_candidates(
            original_query=query,
            intent=intent,
            limit=limit
        )
        from app.services.enrichment_helper import enrich_movie_list
        enriched_results = await enrich_movie_list(results, self.tmdb_service)
        return intent, enriched_results

    async def recommend_movies_from_understanding(
        self, understanding: Any, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Legacy compatibility method for existing tests or endpoints.
        """
        logger.warning("recommend_movies_from_understanding called (legacy method). Mapping to local engine.")
        # Map Gemini understanding to RecommendationIntent
        intent = RecommendationIntent(
            genres=getattr(understanding, "genres", []) or [],
            moods=[getattr(understanding, "mood", "")] if getattr(understanding, "mood", None) else [],
            themes=getattr(understanding, "themes", []) or [],
            similar_movies=getattr(understanding, "reference_movies", []) or [],
            preferred_actors=getattr(understanding, "actors", []) or [],
            preferred_directors=getattr(understanding, "directors", []) or [],
            language=getattr(understanding, "preferred_languages", ["en"])[0] if getattr(understanding, "preferred_languages", None) else None,
            keywords=getattr(understanding, "themes", []) or [],
            legacy_soft_genre=True
        )
        if getattr(understanding, "release_year_constraints", None):
            from app.services.intent_extractor import YearRange
            intent.year_range = YearRange(
                start=getattr(understanding.release_year_constraints, "start_year", None),
                end=getattr(understanding.release_year_constraints, "end_year", None)
            )
        if getattr(understanding, "excluded_genres", None):
            intent.avoid_genres = getattr(understanding, "excluded_genres", [])
            
        return await self.local_retrieval_engine.retrieve_candidates(
            original_query="",
            intent=intent,
            limit=limit
        )

    async def get_recommendations_for_movie(self, movie_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Legacy placeholder.
        """
        logger.info(f"Movie similarity recommendations requested for tmdb_id={movie_id} (limit={limit})")
        intent = RecommendationIntent(similar_movies=[str(movie_id)])
        return await self.local_retrieval_engine.retrieve_candidates(
            original_query="",
            intent=intent,
            limit=limit
        )

    async def get_recommendations_for_user(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Legacy placeholder.
        """
        logger.info(f"User recommendations requested for user_id='{user_id}' (limit={limit})")
        return []

    def _build_semantic_document(self, understanding: Any) -> str:
        """
        Legacy query builder helper.
        """
        parts = []
        if getattr(understanding, "search_intent", None):
            parts.append(f"Intent: {understanding.search_intent}")
        if getattr(understanding, "mood", None):
            parts.append(f"Mood: {understanding.mood}")
        if getattr(understanding, "genres", None):
            parts.append(f"Genres: {', '.join(understanding.genres)}")
        if getattr(understanding, "actors", None):
            parts.append(f"Starring: {', '.join(understanding.actors)}")
        if getattr(understanding, "directors", None):
            parts.append(f"Directed by: {', '.join(understanding.directors)}")
        if getattr(understanding, "reference_movies", None):
            parts.append(f"Like: {', '.join(understanding.reference_movies)}")
        if getattr(understanding, "user_preferences", None):
            parts.append(f"Preferences: {understanding.user_preferences}")
        return " | ".join(parts)

