import re
import asyncio
import threading
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import httpx
from loguru import logger

from app.core.config import settings


class YearConstraints(BaseModel):
    """Schemas representing release year constraints."""
    start_year: Optional[int] = Field(None, description="Minimum release year constraint (inclusive)")
    end_year: Optional[int] = Field(None, description="Maximum release year constraint (inclusive)")
    exact_year: Optional[int] = Field(None, description="Exact release year constraint")


class QueryUnderstandingResult(BaseModel):
    """Schema representing structured query understanding analysis output."""
    search_intent: str = Field(default="unknown", description="Primary query intent (search, recommendation, comparison, etc.)")
    mood: Optional[str] = Field(None, description="Mood or tone extracted from query")
    themes: List[str] = Field(default_factory=list, description="Extracted themes or plot elements")
    genres: List[str] = Field(default_factory=list, description="Target genres matching dataset standards")
    actors: List[str] = Field(default_factory=list, description="Actor names mentioned")
    directors: List[str] = Field(default_factory=list, description="Director names mentioned")
    reference_movies: List[str] = Field(default_factory=list, description="Specific movies referenced in query")
    preferred_languages: List[str] = Field(default_factory=list, description="Languages preferred")
    release_year_constraints: Optional[YearConstraints] = Field(None, description="Extracted release year boundaries")
    excluded_genres: List[str] = Field(default_factory=list, description="Genres explicitly excluded")
    user_preferences: Optional[str] = Field(None, description="Additional custom user requests or constraints")


class GeminiService:
    """
    Service integrating Google AI Studio's Gemini API to perform
    structured natural language query understanding for ChitraAI.
    """
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.GEMINI_MODEL
        self._cache: Dict[str, QueryUnderstandingResult] = {}
        self._cache_lock = threading.Lock()
        logger.info(f"GeminiService initialized with model '{self.model}'. Key configured: {bool(self.api_key)}")

    async def understand_query(self, query: str) -> QueryUnderstandingResult:
        """
        Analyzes the query using Gemini API and parses it to a QueryUnderstandingResult.
        Falls back to local heuristic extraction on API failures or missing key.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return QueryUnderstandingResult()

        # 1. Check local cache
        with self._cache_lock:
            if cleaned_query in self._cache:
                logger.info(f"Cache hit for query: '{cleaned_query}'")
                return self._cache[cleaned_query]

        # 2. Heuristic fallback if API key is not present
        if not self.api_key:
            logger.warning("Gemini API key is not configured. Falling back to local heuristic extraction.")
            result = self._extract_heuristics(cleaned_query)
            with self._cache_lock:
                self._cache[cleaned_query] = result
            return result

        # 3. Call Gemini API
        try:
            result = await self._call_gemini_api(cleaned_query)
            with self._cache_lock:
                self._cache[cleaned_query] = result
            return result
        except Exception as e:
            logger.error(f"Gemini API invocation failed: {e}. Falling back to local heuristics.")
            fallback_result = self._extract_heuristics(cleaned_query)
            return fallback_result

    async def _call_gemini_api(self, query: str) -> QueryUnderstandingResult:
        """Helper to invoke Gemini REST endpoint with backoff retries and JSON Schema."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        
        # Build prompt & generation parameters
        prompt = (
            f"Extract movie search metadata parameters from the user's natural language request: '{query}'. "
            "Follow the output schema exactly."
        )

        request_body = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "search_intent": {
                            "type": "STRING",
                            "description": "User's search intent (e.g. search, recommendation, comparison, unknown)."
                        },
                        "mood": {
                            "type": "STRING",
                            "description": "Extracted mood, e.g. dark, happy, scary."
                        },
                        "themes": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"},
                            "description": "Themes/topics (e.g. space travel, heist, time travel)."
                        },
                        "genres": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"},
                            "description": "Preferred genres (e.g. Action, Comedy, Horror, Sci-Fi)."
                        },
                        "actors": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"},
                            "description": "Actors mentioned."
                        },
                        "directors": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"},
                            "description": "Directors mentioned."
                        },
                        "reference_movies": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"},
                            "description": "Movies mentioned as examples."
                        },
                        "preferred_languages": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"},
                            "description": "Preferred languages, e.g. 'en', 'fr'."
                        },
                        "release_year_constraints": {
                            "type": "OBJECT",
                            "properties": {
                                "start_year": {"type": "INTEGER", "description": "Lower release year bound inclusive"},
                                "end_year": {"type": "INTEGER", "description": "Upper release year bound inclusive"},
                                "exact_year": {"type": "INTEGER", "description": "Exact release year specified"}
                            }
                        },
                        "excluded_genres": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"},
                            "description": "Genres explicitly excluded."
                        },
                        "user_preferences": {
                            "type": "STRING",
                            "description": "Additional custom preferences, e.g. PG-13 only."
                        }
                    },
                    "required": ["search_intent"]
                }
            }
        }

        # Timeout settings (10 seconds total)
        timeout = httpx.Timeout(10.0, connect=3.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(1, 4):
                try:
                    logger.debug(f"Attempting Gemini API request (attempt {attempt}/3)")
                    response = await client.post(url, json=request_body)
                    
                    if response.status_code == 200:
                        data = response.json()
                        text = data["candidates"][0]["content"]["parts"][0]["text"]
                        
                        # Clean markdown code block wraps if returned
                        text = text.strip()
                        if text.startswith("```json"):
                            text = text[7:]
                        if text.endswith("```"):
                            text = text[:-3]
                        text = text.strip()

                        # Parse Pydantic validation
                        return QueryUnderstandingResult.model_validate_json(text)
                    
                    elif response.status_code in (429, 500, 503):
                        backoff = 2 ** attempt
                        logger.warning(f"Gemini API returned status {response.status_code}. Retrying in {backoff}s...")
                        await asyncio.sleep(backoff)
                    else:
                        logger.error(f"Gemini API error status code {response.status_code}: {response.text}")
                        break
                except httpx.RequestError as exc:
                    backoff = 2 ** attempt
                    logger.warning(f"HTTP request error during Gemini call: {exc}. Retrying in {backoff}s...")
                    if attempt == 3:
                        raise exc
                    await asyncio.sleep(backoff)
            
            raise RuntimeError("Gemini API call failed after max retry attempts.")

    def _extract_heuristics(self, query: str) -> QueryUnderstandingResult:
        """
        Regex-based local heuristic fallback extractor to ensure high availability
        when Gemini API is unreachable or key is missing.
        """
        logger.info(f"Running local heuristic parser for query: '{query}'")
        query_lower = query.lower()

        genres = []
        excluded_genres = []
        themes = []
        mood = None
        start_year = None
        end_year = None
        exact_year = None

        # Basic Genre matches
        genre_mapping = {
            "scifi": "Sci-Fi", "sci-fi": "Sci-Fi", "science fiction": "Sci-Fi",
            "horror": "Horror", "scary": "Horror", "spooky": "Horror",
            "comedy": "Comedy", "funny": "Comedy", "hilarious": "Comedy",
            "action": "Action", "thriller": "Thriller", "suspense": "Thriller",
            "drama": "Drama", "romance": "Romance", "love story": "Romance",
            "animation": "Animation", "cartoon": "Animation", "animated": "Animation",
            "documentary": "Documentary", "fantasy": "Fantasy"
        }

        # Exclusion detection (e.g. "no comedy", "not horror", "without romance")
        exclusion_patterns = [
            r"(?:no|not|without|except|excluding)\s+(\w+)"
        ]
        exclusions = []
        for pattern in exclusion_patterns:
            exclusions.extend(re.findall(pattern, query_lower))

        for word in exclusions:
            for key, val in genre_mapping.items():
                if key in word and val not in excluded_genres:
                    excluded_genres.append(val)

        # Standard genre preference inclusion (if not explicitly excluded)
        for key, val in genre_mapping.items():
            if key in query_lower and val not in excluded_genres:
                # Ensure it's not part of an exclusion phrase
                is_excluded = False
                for excl in exclusions:
                    if key in excl:
                        is_excluded = True
                        break
                if not is_excluded and val not in genres:
                    genres.append(val)

        # Basic mood extraction
        if any(w in query_lower for w in ["scary", "spooky", "frightening"]):
            mood = "scary"
        elif any(w in query_lower for w in ["dark", "grim", "somber"]):
            mood = "dark"
        elif any(w in query_lower for w in ["happy", "cheerful", "uplifting", "lighthearted"]):
            mood = "uplifting"

        # Theme extraction
        if "space" in query_lower:
            themes.append("space exploration")
        if "time travel" in query_lower:
            themes.append("time travel")
        if "heist" in query_lower or "robbery" in query_lower:
            themes.append("heist")

        # Year Constraint Parser
        # "after 1990" -> start_year = 1991
        after_match = re.search(r"after\s+(\d{4})", query_lower)
        if after_match:
            start_year = int(after_match.group(1)) + 1
        
        # "before 1980" -> end_year = 1979
        before_match = re.search(r"before\s+(\d{4})", query_lower)
        if before_match:
            end_year = int(before_match.group(1)) - 1
            
        # "in 2012" / "released in 2012"
        in_match = re.search(r"\bin\s+(\d{4})\b", query_lower)
        if in_match and not after_match and not before_match:
            exact_year = int(in_match.group(1))

        year_constraints = None
        if start_year or end_year or exact_year:
            year_constraints = YearConstraints(
                start_year=start_year,
                end_year=end_year,
                exact_year=exact_year
            )

        # Deduce intent
        intent = "search"
        if any(w in query_lower for w in ["compare", "versus", "vs"]):
            intent = "comparison"
        elif any(w in query_lower for w in ["recommend", "suggest"]):
            intent = "recommendation"

        return QueryUnderstandingResult(
            search_intent=intent,
            mood=mood,
            themes=themes,
            genres=genres,
            excluded_genres=excluded_genres,
            release_year_constraints=year_constraints,
            user_preferences="Local Heuristic Fallback Extraction"
        )
