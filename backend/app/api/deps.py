from app.services.embedding_service import EmbeddingService
from app.services.search_service import SearchService
from app.services.recommendation_service import RecommendationService
from app.services.tmdb_service import TMDbService
from app.services.openai_service import OpenAIService
from app.services.local_retrieval import LocalRetrievalEngine
from app.services.intent_extractor import IntentExtractor

# Initialize singleton instances
embedding_service = EmbeddingService()
local_retrieval_engine = LocalRetrievalEngine(embedding_service)
intent_extractor = IntentExtractor()

# Wire local retrieval engine, intent extractor, and TMDB service into services
tmdb_service = TMDbService()
search_service = SearchService(local_retrieval_engine)
recommendation_service = RecommendationService(local_retrieval_engine, intent_extractor, tmdb_service=tmdb_service)
openai_service = OpenAIService()

def get_embedding_service() -> EmbeddingService:
    """Dependency injection helper for embedding generator service."""
    return embedding_service

def get_local_retrieval_engine() -> LocalRetrievalEngine:
    """Dependency injection helper for local retrieval engine."""
    return local_retrieval_engine

def get_intent_extractor() -> IntentExtractor:
    """Dependency injection helper for OpenAI intent extractor."""
    return intent_extractor

def get_search_service() -> SearchService:
    """Dependency injection helper for search engine service."""
    return search_service

def get_recommendation_service() -> RecommendationService:
    """Dependency injection helper for recommendation service."""
    return recommendation_service

def get_tmdb_service() -> TMDbService:
    """Dependency injection helper for TMDb service."""
    return tmdb_service

def get_openai_service() -> OpenAIService:
    """Dependency injection helper for OpenAI service."""
    return openai_service

