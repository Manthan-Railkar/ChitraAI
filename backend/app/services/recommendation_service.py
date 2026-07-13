import math
from typing import Any, List, Dict, Optional
from loguru import logger

from app.vector_db.qdrant import QdrantWrapper
from app.services.embedding_service import EmbeddingService
from app.services.gemini_service import QueryUnderstandingResult


class RecommendationService:
    """
    Hybrid recommendation engine for ChitraAI.
    Combines vector search with structured parameter query building,
    hard pruning (exclusions, year bounds), soft metadata boosting
    (actor/crew/genre matches), and dynamically generated natural-language reasons.
    """
    def __init__(self, qdrant_client: QdrantWrapper, embedding_service: EmbeddingService) -> None:
        self.qdrant = qdrant_client
        self.embedding_service = embedding_service
        logger.info("RecommendationService initialized with Qdrant client and Embedding Service.")

    async def recommend_movies_from_understanding(
        self, understanding: QueryUnderstandingResult, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Processes query parameters, executes filtered semantic retrieval,
        applies hybrid reranking, and constructs recommendation explanations.
        """
        logger.info(f"Generating recommendations from understanding: intent={understanding.search_intent}")

        # 1. Build optimized semantic search document
        search_doc = self._build_semantic_document(understanding)
        logger.info(f"Built semantic search document: '{search_doc}'")

        # 2. Get Qdrant client connection
        if not self.qdrant.client:
            logger.warning("Qdrant client not initialized. Connecting...")
            if not self.qdrant.connect():
                logger.error("Could not connect to Qdrant. Returning empty recommendations.")
                return []

        # 3. Compute vector embedding
        try:
            query_vector = self.embedding_service.encode_single(search_doc)
        except Exception as e:
            logger.error(f"Failed to generate search document embedding: {e}")
            return []

        # 4. Perform vector search (retrieve a large candidate pool for robust filtering)
        candidate_limit = max(100, limit * 10)
        try:
            hits = self.qdrant.search(
                query_vector=query_vector.tolist(),
                limit=candidate_limit
            )
            logger.info(f"Retrieved {len(hits)} raw candidates from Qdrant.")
        except Exception as e:
            logger.error(f"Vector search retrieval failed: {e}")
            return []

        if not hits:
            return []

        # 5. Apply Hard Filters, Soft Boosting, Reranking & Reason Generation
        recommended_movies = []
        for hit in hits:
            payload = hit["payload"] or {}
            
            # --- HARD FILTERS ---
            
            # 5a. Excluded Genres (case-insensitive check)
            if understanding.excluded_genres and "genres" in payload:
                movie_genres_lower = [g.lower() for g in payload.get("genres", [])]
                has_exclusion = any(excl.lower() in movie_genres_lower for excl in understanding.excluded_genres)
                if has_exclusion:
                    continue  # Prune candidate

            # 5b. Release Year Constraints
            if understanding.release_year_constraints and "release_year" in payload:
                ry = payload.get("release_year")
                if ry is not None:
                    constraints = understanding.release_year_constraints
                    # Exact Year Check
                    if constraints.exact_year is not None and ry != constraints.exact_year:
                        continue
                    # Start Year Bound Check
                    if constraints.start_year is not None and ry < constraints.start_year:
                        continue
                    # End Year Bound Check
                    if constraints.end_year is not None and ry > constraints.end_year:
                        continue

            # --- SOFT BOOSTING ---
            
            # Base semantic score (Cosine similarity typically range [0.0, 1.0])
            semantic_score = float(hit["score"])
            boost = 0.0

            matched_genres = []
            matched_actors = []
            matched_directors = []
            matched_themes = []

            # Genre matches: +0.03 per match
            if understanding.genres and "genres" in payload:
                movie_genres = payload.get("genres", [])
                for g in understanding.genres:
                    if any(g.lower() == mg.lower() for mg in movie_genres):
                        boost += 0.03
                        # Find correct capitalized name from payload
                        matched_g = next((mg for mg in movie_genres if g.lower() == mg.lower()), g)
                        matched_genres.append(matched_g)

            # Actor matches: +0.05 per match
            if understanding.actors and "cast" in payload:
                movie_cast = payload.get("cast", [])
                for actor in understanding.actors:
                    if any(actor.lower() in mc.lower() for mc in movie_cast):
                        boost += 0.05
                        matched_a = next((mc for mc in movie_cast if actor.lower() in mc.lower()), actor)
                        matched_actors.append(matched_a)

            # Director matches: +0.05 per match
            if understanding.directors and "directors" in payload:
                movie_directors = payload.get("directors", [])
                for director in understanding.directors:
                    if any(director.lower() in md.lower() for md in movie_directors):
                        boost += 0.05
                        matched_d = next((md for md in movie_directors if director.lower() in md.lower()), director)
                        matched_directors.append(matched_d)

            # Theme/Keyword matches: +0.02 per match
            if understanding.themes:
                title = payload.get("title", "").lower()
                overview = payload.get("overview", "").lower()
                keywords = [k.lower() for k in payload.get("keywords", [])]
                for theme in understanding.themes:
                    theme_l = theme.lower()
                    if theme_l in title or theme_l in overview or any(theme_l in kw for kw in keywords):
                        boost += 0.02
                        matched_themes.append(theme)

            # Apply boost and cap semantic similarity at 1.0
            boosted_semantic_score = min(1.0, semantic_score + boost)

            # --- HYBRID RERANKING ---
            
            rating_value = payload.get("rating_value")
            popularity = payload.get("popularity")
            vote_count = payload.get("vote_count")

            rating_val = float(rating_value) if rating_value is not None else 5.0
            pop_val = float(popularity) if popularity is not None else 0.0
            votes_val = int(vote_count) if vote_count is not None else 0

            # Scale/Normalize
            score_rating = rating_val / 10.0
            score_popularity = min(1.0, math.log1p(pop_val) / 5.0)
            score_votes = min(1.0, math.log1p(votes_val) / 15.0)

            # Combined Score Formula (weights sum to 1.0)
            reranked_score = (
                0.6 * boosted_semantic_score +
                0.2 * score_rating +
                0.1 * score_popularity +
                0.1 * score_votes
            )

            # --- RECOMMENDATION REASONS GENERATION ---
            
            reasons = []
            if matched_genres:
                reasons.append(f"matches preferred genre(s) ({', '.join(matched_genres)})")
            if matched_actors:
                reasons.append(f"features actor(s) you like ({', '.join(matched_actors)})")
            if matched_directors:
                reasons.append(f"is directed by your requested director(s) ({', '.join(matched_directors)})")
            if matched_themes:
                reasons.append(f"aligns with theme(s) '{', '.join(matched_themes)}'")

            if reasons:
                recommendation_reason = "Recommended because it " + " and ".join(reasons) + "."
            else:
                recommendation_reason = "Recommended due to high semantic relevance to your request."

            movie_data = {
                "id": str(hit["id"]),
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
                "semantic_score": semantic_score,
                "boosted_semantic_score": round(boosted_semantic_score, 4),
                "reranked_score": round(reranked_score, 4),
                "recommendation_reason": recommendation_reason
            }
            recommended_movies.append(movie_data)

        # Sort descending by reranked_score and slice to limit
        recommended_movies.sort(key=lambda x: x["reranked_score"], reverse=True)
        final_recommendations = recommended_movies[:limit]

        logger.info(f"Completed recommendations. Returning top {len(final_recommendations)} movies.")
        return final_recommendations

    def _build_semantic_document(self, understanding: QueryUnderstandingResult) -> str:
        """
        Converts structured query understanding parameters into an optimized,
        semantic search query document to be encoded by the embedding model.
        """
        parts = []
        if understanding.search_intent and understanding.search_intent != "unknown":
            parts.append(f"Intent: {understanding.search_intent}")
        if understanding.mood:
            parts.append(f"Mood: {understanding.mood}")
        if understanding.genres:
            parts.append(f"Genres: {', '.join(understanding.genres)}")
        if understanding.themes:
            parts.append(f"Themes: {', '.join(understanding.themes)}")
        if understanding.actors:
            parts.append(f"Starring: {', '.join(understanding.actors)}")
        if understanding.directors:
            parts.append(f"Directed by: {', '.join(understanding.directors)}")
        if understanding.reference_movies:
            parts.append(f"Like: {', '.join(understanding.reference_movies)}")
        if understanding.user_preferences:
            parts.append(f"Preferences: {understanding.user_preferences}")

        return ". ".join(parts)

    async def get_recommendations_for_movie(self, movie_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Legacy mock placeholder. Recommends movies similar to a given movie ID.
        """
        logger.info(f"Mock recommendation requested for movie_id={movie_id} with limit={limit}")
        return [
            {
                "id": "mock-3",
                "title": "The Dark Knight",
                "overview": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham.",
                "score": 0.85
            }
        ]

    async def get_recommendations_for_user(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Legacy mock placeholder. Recommends movies based on user profiles.
        """
        logger.info(f"Mock recommendation requested for user_id='{user_id}' with limit={limit}")
        return [
            {
                "id": "mock-4",
                "title": "The Matrix",
                "overview": "A computer hacker learns from mysterious rebels about the true nature of his reality.",
                "score": 0.91
            }
        ]
