import sys
import os
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import asyncio
import time
from app.services.local_retrieval import LocalRetrievalEngine
from app.services.embedding_service import EmbeddingService
from app.services.tmdb_service import TMDbService
from app.services.recommendation_service import RecommendationService
from app.services.intent_extractor import IntentExtractor

async def main():
    print("Initializing services for verification...")
    # Load env variables and config
    from app.core.config import settings
    
    # 1. Initialize dependencies
    embedding_service = EmbeddingService()
    local_engine = LocalRetrievalEngine(embedding_service=embedding_service)
    local_engine.initialize()
    
    # We will use the real Groq API or OpenAI depending on what is set up
    intent_extractor = IntentExtractor()
    tmdb_service = TMDbService()
    
    recommend_service = RecommendationService(
        local_retrieval_engine=local_engine,
        intent_extractor=intent_extractor,
        tmdb_service=tmdb_service
    )
    
    queries = [
        "Interstellar",
        "The Dark Knight",
        "Fight Club",
        "Oppenheimer",
        "The Godfather",
        "Best crime movies",
        "Best sci-fi movies",
        "Movies like Interstellar",
        "Funny family movies"
    ]
    
    print("\nStarting enrichment verification for 9 target queries...\n")
    for q in queries:
        print(f"===========================================================")
        print(f"QUERY: '{q}'")
        print(f"===========================================================")
        start_time = time.perf_counter()
        
        try:
            intent, results = await recommend_service.recommend_movies_from_query(q, limit=3)
            elapsed = (time.perf_counter() - start_time) * 1000
            print(f"Response Latency: {elapsed:.2f} ms")
            print(f"Returned {len(results)} movies:")
            
            for idx, movie in enumerate(results):
                print(f"\n  Rank {idx + 1}: {movie.get('title')} ({movie.get('release_year')})")
                print(f"    - TMDb ID: {movie.get('tmdb_id')}")
                print(f"    - IMDb ID: {movie.get('imdb_id')}")
                print(f"    - Original Language: {movie.get('original_language')}")
                print(f"    - Overview: {movie.get('overview')[:100]}...")
                print(f"    - Genres: {movie.get('genres')}")
                print(f"    - Director: {movie.get('directors')}")
                print(f"    - Writers: {movie.get('writers')}")
                print(f"    - Producer: {movie.get('producer')}")
                print(f"    - Composer: {movie.get('composer')}")
                print(f"    - Cinematographer: {movie.get('cinematographer')}")
                print(f"    - Poster Path: {movie.get('poster_path')}")
                print(f"    - Backdrop Path: {movie.get('backdrop_path')}")
                print(f"    - Logo URL: {movie.get('logo_url')}")
                print(f"    - Budget: ${movie.get('budget'):,}" if movie.get('budget') else "    - Budget: N/A")
                print(f"    - Revenue: ${movie.get('revenue'):,}" if movie.get('revenue') else "    - Revenue: N/A")
                print(f"    - Trailer Link: {movie.get('trailer_url')}")
                print(f"    - Trailer Name: {movie.get('trailer_name')}")
                print(f"    - Trailer Type: {movie.get('trailer_type')}")
                print(f"    - Streaming Providers: {movie.get('streaming_providers')}")
                print(f"    - Confidence: {movie.get('confidence_score')}%")
                print(f"    - Reason: {movie.get('recommendation_reason')}")
                
                # Check top 5 similar
                similar = [f"{sm.get('title')} ({sm.get('release_year')})" for sm in movie.get('similar_movies', [])]
                print(f"    - Top 5 Similar: {similar}")
                
                # Check top 5 recommended
                recs = [f"{rm.get('title')} ({rm.get('release_year')})" for rm in movie.get('recommended_movies', [])]
                print(f"    - Top 5 Recommended: {recs}")
                
        except Exception as e:
            print(f"Error executing query '{q}': {e}")
        print()

if __name__ == "__main__":
    asyncio.run(main())
