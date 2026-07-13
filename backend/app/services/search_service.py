import math
from typing import Any, List, Dict
from loguru import logger
from app.vector_db.qdrant import QdrantWrapper
from app.services.embedding_service import EmbeddingService

class SearchService:
    """
    Core search engine service for ChitraAI.
    Computes query embeddings, performs nearest-neighbor retrieval in Qdrant,
    and applies a custom hybrid reranking formula based on metadata authority.
    """
    def __init__(self, qdrant_client: QdrantWrapper, embedding_service: EmbeddingService) -> None:
        self.qdrant = qdrant_client
        self.embedding_service = embedding_service
        logger.info("SearchService initialized with Qdrant and Embedding services.")

    async def search_movies(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Performs a semantic search and custom metadata-based reranking.
        
        Args:
            query: The natural language search query.
            limit: The final number of ranked movies to return.
            
        Returns:
            List of movie result dicts, sorted by reranked_score.
        """
        logger.info(f"Executing semantic search for query: '{query}' (limit={limit})")
        
        # 1. Ensure Qdrant is connected
        if not self.qdrant.client:
            logger.warning("Qdrant client not initialized. Attempting connection...")
            if not self.qdrant.connect():
                logger.error("Could not connect to Qdrant. Returning empty search results.")
                return []

        # 2. Generate Query Embedding
        try:
            query_vector = self.embedding_service.encode_single(query)
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return []

        # 3. Retrieve Nearest Neighbors from Qdrant
        # We retrieve candidates matching max(50, limit * 5) to allow robust reranking
        candidate_limit = max(50, limit * 5)
        try:
            hits = self.qdrant.search(
                query_vector=query_vector.tolist(),
                limit=candidate_limit
            )
            logger.info(f"Retrieved {len(hits)} raw candidates from Qdrant collection '{self.qdrant.collection_name}'.")
        except Exception as e:
            logger.error(f"Qdrant vector search failed: {e}")
            return []

        if not hits:
            return []

        # 4. Hybrid Reranking
        reranked_results = []
        for hit in hits:
            payload = hit["payload"] or {}
            
            # Extract ranking signals
            rating_value = payload.get("rating_value")
            popularity = payload.get("popularity")
            vote_count = payload.get("vote_count")

            # Handle nulls
            rating_val = float(rating_value) if rating_value is not None else 5.0
            pop_val = float(popularity) if popularity is not None else 0.0
            votes_val = int(vote_count) if vote_count is not None else 0

            # Scale/Normalize components
            # Cosine similarity hit["score"] is typically in [0, 1] range for normalized vectors
            score_semantic = float(hit["score"])
            
            # Scale rating to [0.1, 1.0] range
            score_rating = rating_val / 10.0
            
            # Log-scale popularity to [0.0, 1.0] range, capped at 1.0
            score_popularity = min(1.0, math.log1p(pop_val) / 5.0)
            
            # Log-scale votes to [0.0, 1.0] range, capped at 1.0
            score_votes = min(1.0, math.log1p(votes_val) / 15.0)

            # Combined Score Formula (weights sum to 1.0)
            # 60% semantic similarity, 20% IMDb rating, 10% TMDb popularity, 10% vote count
            reranked_score = (
                0.6 * score_semantic +
                0.2 * score_rating +
                0.1 * score_popularity +
                0.1 * score_votes
            )

            # Map to final metadata output structure
            movie_data = {
                "id": str(hit["id"]),
                "title": payload.get("title", "Unknown Movie"),
                "original_title": payload.get("original_title"),
                "overview": payload.get("overview"),
                "genres": payload.get("genres", []),
                "directors": payload.get("directors", []),
                "release_year": payload.get("release_year"),
                "rating_value": payload.get("rating_value"),
                "popularity": payload.get("popularity"),
                "vote_count": payload.get("vote_count"),
                "poster_path": payload.get("poster_path"),
                "semantic_score": score_semantic,
                "reranked_score": round(reranked_score, 4)
            }
            reranked_results.append(movie_data)

        # 5. Sort descending by reranked_score and slice to limit
        reranked_results.sort(key=lambda x: x["reranked_score"], reverse=True)
        final_results = reranked_results[:limit]
        
        logger.info(f"Completed search & reranking. Returning top {len(final_results)} movies.")
        return final_results
