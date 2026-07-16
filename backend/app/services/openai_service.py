import json
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional
from loguru import logger
from app.core.model_manager import ModelManager
from app.core.config import settings


class OpenAICache:
    """Thread-safe LRU cache for OpenAI fallback recommendations."""
    def __init__(self, maxsize: int = 100) -> None:
        self.cache: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
        self.maxsize = maxsize
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[List[Dict[str, Any]]]:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def set(self, key: str, value: List[Dict[str, Any]]) -> None:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)


class OpenAIService:
    """
    Service for generating movie recommendations using OpenAI gpt-4o-mini as a fallback.
    Caches results in a thread-safe LRU cache.
    """
    def __init__(self) -> None:
        self._cache = OpenAICache(maxsize=100)
        logger.info("OpenAIService initialized with thread-safe OpenAI response cache.")

    async def get_fallback_recommendations(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Invokes OpenAI model to generate structured movie recommendations.
        """
        cache_key = f"{query}_{limit}"
        cached_val = self._cache.get(cache_key)
        if cached_val is not None:
            logger.info(f"[OpenAI Cache] HIT for query: '{query}'")
            return cached_val

        logger.info(f"[OpenAI Cache] MISS for query: '{query}'. Calling OpenAI API fallback...")
        
        # Retrieve client from ModelManager
        client = ModelManager.get_openai_client()
        
        system_prompt = (
            "You are ChitraAI, a professional cinematic movie recommendation assistant. "
            "Your task is to recommend high-quality, relevant movies matching the user's search query, preferences, mood, or themes. "
            "You must return the recommendations as a JSON object containing a 'results' key which holds an array of movie objects. "
            "Each movie object MUST conform to this schema:\n"
            "{\n"
            '  "title": "Movie Title (required)",\n'
            '  "release_year": 2010 (integer release year, required),\n'
            '  "genres": ["Action", "Sci-Fi"] (array of strings, required),\n'
            '  "overview": "A brief overview plot summary... (required)",\n'
            '  "rating_value": 8.2 (float IMDb/TMDb rating between 1 and 10, required),\n'
            '  "popularity": 120.5 (float popularity, required),\n'
            '  "vote_count": 5000 (integer vote count, required),\n'
            '  "recommendation_reason": "Provide a concise explanation of why this movie matches the request (required)"\n'
            "}\n"
            "Provide exactly the number of recommendations requested by the limit parameter. Do not wrap in markdown or add anything else. "
            "Output MUST be valid, parseable JSON."
        )

        user_prompt = f"User Request: '{query}'\nLimit: {limit} movie recommendations."

        try:
            response = await client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=2500
            )
            
            content = response.choices[0].message.content or "{}"
            data = json.loads(content)
            results = data.get("results", [])
            
            processed_results = []
            for idx, r in enumerate(results):
                movie = {
                    "imdb_id": None,
                    "tmdb_id": None,
                    "movielens_id": None,
                    "wiki_page": None,
                    "title": r.get("title", "Unknown Movie"),
                    "original_title": r.get("title"),
                    "overview": r.get("overview", ""),
                    "plot_summary": None,
                    "genres": r.get("genres", []),
                    "cast": [],
                    "directors": [],
                    "writers": [],
                    "runtime_minutes": None,
                    "release_year": r.get("release_year"),
                    "rating_value": r.get("rating_value"),
                    "vote_count": r.get("vote_count", 0),
                    "popularity": r.get("popularity", 0.0),
                    "production_companies": [],
                    "languages": ["en"],
                    "keywords": [],
                    "source_dataset": "openai_fallback",
                    "poster_path": None,
                    "backdrop_path": None,
                    "trailer_url": None,
                    "streaming_providers": [],
                    "collection_name": None,
                    "certification": None,
                    "semantic_score": round(1.0 - (idx * 0.05), 4),
                    "boosted_semantic_score": round(1.0 - (idx * 0.05), 4),
                    "reranked_score": round(1.0 - (idx * 0.05), 4),
                    "recommendation_reason": r.get("recommendation_reason", "Recommended based on search preferences.")
                }
                processed_results.append(movie)

            self._cache.set(cache_key, processed_results)
            return processed_results
            
        except Exception as e:
            logger.error(f"OpenAI fallback recommendation failed: {e}")
            raise e
