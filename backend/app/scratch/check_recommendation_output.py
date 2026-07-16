import sys
from pathlib import Path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import asyncio
from app.services.local_retrieval import LocalRetrievalEngine
from app.services.embedding_service import EmbeddingService
from app.services.tmdb_service import TMDbService
from app.services.recommendation_service import RecommendationService
from app.services.intent_extractor import IntentExtractor

async def main():
    embedding_service = EmbeddingService()
    local_engine = LocalRetrievalEngine(embedding_service=embedding_service)
    local_engine.initialize()
    
    intent_extractor = IntentExtractor()
    tmdb_service = TMDbService()
    
    recommend_service = RecommendationService(
        local_retrieval_engine=local_engine,
        intent_extractor=intent_extractor,
        tmdb_service=tmdb_service
    )
    
    intent, results = await recommend_service.recommend_movies_from_query("Oppenheimer", limit=3)
    print("\nKeys in first movie:")
    first = results[0]
    for k, v in first.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    asyncio.run(main())
