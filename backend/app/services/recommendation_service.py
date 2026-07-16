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

        # 2. Retrieve candidates locally
        results = await self.local_retrieval_engine.retrieve_candidates(
            original_query=query,
            intent=intent,
            limit=limit
        )
        
        return intent, results

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

