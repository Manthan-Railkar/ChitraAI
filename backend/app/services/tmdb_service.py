import asyncio
import time
import json
import hashlib
from typing import Any, Dict, Optional, List
from loguru import logger
import httpx
from app.core.config import settings
from app.services.tmdb_cache import TMDbCacheManager


class TMDbService:
    """
    Service for interacting with the TMDb (The Movie Database) API to fetch metadata asynchronously.
    Integrates local caching, concurrency semaphores, rate-limit backoff, and automatic retry mechanisms.
    """
    BASE_URL = "https://api.themoviedb.org/3"

    def __init__(self, cache_manager: Optional[TMDbCacheManager] = None) -> None:
        self.api_key = settings.TMDB_API_KEY
        self.cache = cache_manager or TMDbCacheManager()
        
        # Concurrency controller to prevent overloading the system or API limits
        self.semaphore = asyncio.Semaphore(settings.TMDB_SEMAPHORE_LIMIT)
        self._client: Optional[httpx.AsyncClient] = None
        
        logger.info(f"TMDbService initialized. API Key Configured: {bool(self.api_key)}")

    def get_client(self) -> httpx.AsyncClient:
        """Retrieves or initializes a persistent AsyncClient."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.TIMEOUT_SECONDS, connect=settings.TIMEOUT_SECONDS)
            )
        return self._client

    async def close(self) -> None:
        """Closes the persistent AsyncClient connection pool."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.info("TMDbService persistent HTTP client closed.")

    def _get_headers_and_params(self) -> tuple[Dict[str, str], Dict[str, str]]:
        """Resolves headers and parameters depending on Bearer Token vs API key."""
        headers = {}
        params = {}
        
        # Check if the API key looks like a JWT/v4 Bearer Token
        if self.api_key.startswith("ey"):
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            params["api_key"] = self.api_key
            
        return headers, params

    async def _make_request(
        self, client: httpx.AsyncClient, endpoint: str, extra_params: Optional[Dict[str, Any]] = None, max_retries: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Makes an asynchronous request to the TMDb API with exponential backoff and rate-limit handling."""
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        headers, params = self._get_headers_and_params()
        if extra_params:
            params.update(extra_params)

        retries_limit = max_retries if max_retries is not None else settings.MAX_RETRIES

        for attempt in range(1, retries_limit + 1):
            async with self.semaphore:
                try:
                    response = await client.get(url, headers=headers, params=params)
                    
                    # 429 Too Many Requests
                    if response.status_code == 429:
                        retry_after = response.headers.get("Retry-After")
                        sleep_seconds = float(retry_after) if retry_after else (0.5 * attempt)
                        logger.warning(
                            f"[TMDb API] Rate limit hit (429) on {endpoint}. "
                            f"Sleeping for {sleep_seconds:.1f}s (Attempt {attempt}/{retries_limit})."
                        )
                        await asyncio.sleep(sleep_seconds)
                        continue
                        
                    # Other status errors
                    response.raise_for_status()
                    return response.json()
                    
                except httpx.HTTPStatusError as e:
                    # If 404 (Not Found) or 401/403 (Unauthorized/Forbidden), do not retry
                    if e.response.status_code == 404:
                        logger.debug(f"[TMDb API] Resource not found (404) for {endpoint}")
                        return None
                    if e.response.status_code in (401, 403):
                        logger.error(
                            f"[TMDb API] Authentication error {e.response.status_code} on {endpoint}. "
                            "Please check that your TMDB_API_KEY is valid."
                        )
                        return None
                        
                    logger.warning(
                        f"[TMDb API] HTTP error {e.response.status_code} on {endpoint}. "
                        f"Attempt {attempt}/{retries_limit}. Retry in {0.5 * attempt}s."
                    )
                    await asyncio.sleep(0.5 * attempt)
                    
                except (httpx.RequestError, asyncio.TimeoutError) as e:
                    logger.warning(
                        f"[TMDb API] Request failed: {e} on {endpoint}. "
                        f"Attempt {attempt}/{retries_limit}. Retry in {0.5 * attempt}s."
                    )
                    await asyncio.sleep(0.5 * attempt)
                    
        logger.error(f"[TMDb API] Max retries reached ({retries_limit}) for {endpoint}.")
        return None

    async def fetch_movie_details(self, tmdb_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves full movie details (videos, watch/providers, release_dates, keywords, credits)
        from cache or the TMDb API.
        """
        # 1. Check local cache
        cached = self.cache.get_movie_details(tmdb_id)
        if cached is not None:
            return cached

        # If API key is empty/not set (e.g. running mock or tests without key), return placeholder or None
        if not self.api_key:
            logger.warning("[TMDb API] No API Key set. Skipping API request.")
            return None

        # 2. Fetch from API
        endpoint = f"movie/{tmdb_id}"
        extra_params = {
            "append_to_response": "videos,watch/providers,release_dates,keywords,credits,similar,recommendations,images"
        }
        
        client = self.get_client()
        response = await self._make_request(client, endpoint, extra_params)
            
        if response:
            # 3. Store in local cache
            self.cache.save_movie_details(tmdb_id, response)
            
        return response

    async def fetch_tmdb_id_by_imdb(self, imdb_id: str) -> Optional[int]:
        """
        Resolves an IMDb ID (ttxxxxxxx) to a TMDb ID using the /find endpoint.
        Uses local cache to prevent duplicate queries.
        """
        # 1. Check local cache
        cached_id = self.cache.get_tmdb_id_by_imdb(imdb_id)
        # Note: cached_id could be 0 or None if previously checked and not found
        if cached_id is not None:
            return cached_id if cached_id > 0 else None

        if not self.api_key:
            logger.warning("[TMDb API] No API Key set. Skipping IMDb resolve.")
            return None

        # 2. Fetch from API
        endpoint = f"find/{imdb_id}"
        extra_params = {"external_source": "imdb_id"}
        
        client = self.get_client()
        response = await self._make_request(client, endpoint, extra_params)
            
        tmdb_id = None
        if response:
            results = response.get("movie_results", [])
            if results:
                tmdb_id = results[0].get("id")
                
        # 3. Store in cache (store 0 to represent a checked-but-unmapped ID, to avoid re-querying)
        self.cache.save_imdb_mapping(imdb_id, tmdb_id or 0)
        
        return tmdb_id

    async def search_movie_by_title(self, title: str, year: Optional[int] = None) -> Optional[int]:
        """
        Searches TMDb for a movie by its title (and optional release year)
        and returns the first matching TMDb ID.
        """
        if not self.api_key:
            logger.warning("[TMDb API] No API Key set. Skipping movie search.")
            return None

        endpoint = "search/movie"
        extra_params = {"query": title}
        if year:
            extra_params["primary_release_year"] = year

        client = self.get_client()
        response = await self._make_request(client, endpoint, extra_params)

        if response:
            results = response.get("results", [])
            if results:
                return results[0].get("id")
        return None

    async def discover_movies(self, extra_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Queries the TMDb /discover/movie API with the provided parameters,
        handling caching and concurrency semaphores.
        """
        # Convert params to a sorted JSON string to make it deterministic
        sorted_params_str = json.dumps(extra_params, sort_keys=True)
        query_hash = hashlib.md5(sorted_params_str.encode("utf-8")).hexdigest()

        # Check local cache
        cached = self.cache.get_discover_results(query_hash)
        if cached is not None:
            logger.info(f"[TMDb API] Discover results fetched from cache (hash: {query_hash})")
            return cached

        if not self.api_key:
            logger.warning("[TMDb API] No API Key set. Skipping discover request.")
            return None

        # Fetch from API
        endpoint = "discover/movie"
        client = self.get_client()
        response = await self._make_request(client, endpoint, extra_params)

        if response:
            self.cache.save_discover_results(query_hash, response)

        return response

    async def resolve_keyword_id(self, keyword: str) -> Optional[int]:
        """Resolves a keyword string to its TMDb keyword ID using cached search."""
        key = f"kw:{keyword.lower()}"
        cached_id = self.cache.get_tmdb_id_by_imdb(key)
        if cached_id is not None:
            return cached_id if cached_id > 0 else None

        if not self.api_key:
            return None

        endpoint = "search/keyword"
        extra_params = {"query": keyword}
        client = self.get_client()
        response = await self._make_request(client, endpoint, extra_params)
        
        kw_id = None
        if response:
            results = response.get("results", [])
            if results:
                kw_id = results[0].get("id")
                
        self.cache.save_imdb_mapping(key, kw_id or 0)
        return kw_id

    async def resolve_person_id(self, name: str) -> Optional[int]:
        """Resolves a person's name to their TMDb person ID using cached search."""
        key = f"person:{name.lower()}"
        cached_id = self.cache.get_tmdb_id_by_imdb(key)
        if cached_id is not None:
            return cached_id if cached_id > 0 else None

        if not self.api_key:
            return None

        endpoint = "search/person"
        extra_params = {"query": name}
        client = self.get_client()
        response = await self._make_request(client, endpoint, extra_params)
        
        person_id = None
        if response:
            results = response.get("results", [])
            if results:
                person_id = results[0].get("id")
                
        self.cache.save_imdb_mapping(key, person_id or 0)
        return person_id
