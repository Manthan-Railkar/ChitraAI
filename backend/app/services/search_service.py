from typing import Any, List, Dict
from loguru import logger
from app.services.local_retrieval import LocalRetrievalEngine
from app.services.intent_extractor import RecommendationIntent

class SearchService:
    """
    Core search engine service for ChitraAI.
    Leverages local retrieval engine to perform semantic search over TMDb dataset.
    """
    def __init__(self, local_retrieval_engine: LocalRetrievalEngine) -> None:
        self.local_retrieval_engine = local_retrieval_engine
        logger.info("SearchService initialized with local retrieval engine.")

    async def search_movies(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Performs a local semantic search over the TMDb dataset.
        
        Args:
            query: The natural language search query.
            limit: The final number of ranked movies to return.
            
        Returns:
            List of movie result dicts, sorted by retrieval_score.
        """
        logger.info(f"Executing semantic search for query: '{query}' (limit={limit})")
        
        # For a standard search, we use an empty intent (so no hard filters are applied,
        # only pure semantic search + metadata popularity/rating scoring).
        intent = RecommendationIntent()
        
        try:
            results = await self.local_retrieval_engine.retrieve_candidates(
                original_query=query,
                intent=intent,
                limit=limit
            )
            return results
        except Exception as e:
            logger.error(f"Local semantic search failed: {e}")
            return []
