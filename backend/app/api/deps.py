from app.vector_db.qdrant import QdrantWrapper
from app.services.embedding_service import EmbeddingService
from app.services.search_service import SearchService
from app.services.gemini_service import GeminiService
from app.services.recommendation_service import RecommendationService
from app.services.tmdb_service import TMDbService

# Initialize singleton instances
qdrant_wrapper = QdrantWrapper()
embedding_service = EmbeddingService()
search_service = SearchService(qdrant_wrapper, embedding_service)
gemini_service = GeminiService()
recommendation_service = RecommendationService(qdrant_wrapper, embedding_service)
tmdb_service = TMDbService()

def get_qdrant_wrapper() -> QdrantWrapper:
    """Dependency injection helper for Qdrant client wrapper."""
    return qdrant_wrapper

def get_embedding_service() -> EmbeddingService:
    """Dependency injection helper for embedding generator service."""
    return embedding_service

def get_search_service() -> SearchService:
    """Dependency injection helper for search engine service."""
    return search_service

def get_gemini_service() -> GeminiService:
    """Dependency injection helper for Gemini query understanding service."""
    return gemini_service

def get_recommendation_service() -> RecommendationService:
    """Dependency injection helper for recommendation service."""
    return recommendation_service

def get_tmdb_service() -> TMDbService:
    """Dependency injection helper for TMDb service."""
    return tmdb_service
