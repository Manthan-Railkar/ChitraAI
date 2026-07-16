import json
import re
import time
import threading
import difflib
from pathlib import Path
from typing import List, Optional, Any, Dict
from collections import OrderedDict
from pydantic import BaseModel, Field, AliasChoices
from loguru import logger
from app.core.config import settings
from app.core.model_manager import ModelManager


class YearRange(BaseModel):
    start: Optional[int] = Field(None, description="Starting year of release (inclusive)")
    end: Optional[int] = Field(None, description="Ending year of release (inclusive)")


class RecommendationIntent(BaseModel):
    intent: str = Field(default="recommendation", description="Primary query intent: 'movie_lookup', 'recommendation', or 'general_search'")
    ranking_mode: str = Field(default="default", description="Ranking profile: 'best', 'similar_movie', 'mood', or 'default'")
    genres: List[str] = Field(default_factory=list, description="Target genres (e.g. Action, Thriller, Sci-Fi)")
    moods: List[str] = Field(default_factory=list, validation_alias=AliasChoices("moods", "mood"), description="Target moods or tones (e.g. dark, suspenseful, feel-good)")
    themes: List[str] = Field(default_factory=list, description="Target themes or topics (e.g. mind-bending, heist, time travel)")
    similar_movies: List[str] = Field(default_factory=list, description="Specific movies mentioned to be similar to")
    preferred_actors: List[str] = Field(default_factory=list, validation_alias=AliasChoices("preferred_actors", "actors"), description="Preferred actors mentioned")
    preferred_directors: List[str] = Field(default_factory=list, validation_alias=AliasChoices("preferred_directors", "directors"), description="Preferred directors mentioned")
    language: Optional[str] = Field(None, description="ISO-639-1 language code or language name")
    year_range: Optional[YearRange] = Field(None, description="Release year range constraints")
    runtime: Optional[int] = Field(None, description="Preferred maximum runtime in minutes")
    keywords: List[str] = Field(default_factory=list, description="Key plot elements, keywords, or topics")
    avoid_movies: List[str] = Field(default_factory=list, description="Movies the user explicitly wants to avoid")
    avoid_genres: List[str] = Field(default_factory=list, description="Genres the user explicitly wants to avoid")
    legacy_soft_genre: bool = Field(False, description="Internal flag for backward compatibility to treat genre as soft preference")
    release_year: Optional[int] = Field(None, description="Exact release year constraint")
    exclusions: List[str] = Field(default_factory=list, description="Explicit exclusions (genres, themes, actors, or movie titles the user wants to avoid)")


class IntentCache:
    """Thread-safe LRU cache for extracted recommendation intents."""
    def __init__(self, maxsize: int = 100) -> None:
        self.cache: OrderedDict[str, RecommendationIntent] = OrderedDict()
        self.maxsize = maxsize
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[RecommendationIntent]:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def set(self, key: str, value: RecommendationIntent) -> None:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)


class LLMIntentExtractor:
    """
    Base abstraction layer for executing intent extraction completions on any LLM provider.
    Supports groq, openai, gemini, and future providers.
    """
    def __init__(self) -> None:
        self.provider = settings.LLM_PROVIDER.lower()
        self.model = settings.LLM_MODEL
        logger.info(f"LLMIntentExtractor initialized with provider: {self.provider}, model: {self.model}")

    async def extract_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Calls the configured LLM API provider to retrieve completions."""
        client = ModelManager.get_openai_client()
        
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=256
        )
        return response.choices[0].message.content or "{}"


class IntentExtractor(LLMIntentExtractor):
    """
    Service for parsing natural language movie queries into structured intent.
    Uses local routing, LLM extraction with provider abstraction, and robust fallback heuristics.
    """
    def __init__(self) -> None:
        super().__init__()
        self._cache = IntentCache(maxsize=100)
        
        # Load canonical movie titles for local routing bypass
        self.movie_titles = []
        try:
            processed_dir = Path(settings.PROCESSED_DATA_DIR)
            tmdb_parquet_path = processed_dir / "canonical" / "tmdb_canonical.parquet"
            if tmdb_parquet_path.exists():
                import polars as pl
                df = pl.read_parquet(tmdb_parquet_path)
                self.movie_titles = [t for t in df.select("title").to_series().to_list() if t is not None]
                logger.info(f"Loaded {len(self.movie_titles)} movie titles for local intent routing.")
        except Exception as e:
            logger.warning(f"Could not load movie titles for local routing: {e}")

    async def extract_intent(self, query: str) -> RecommendationIntent:
        """
        Extracts recommendation intent from query using configured LLM or local shortcuts.
        Uses a local thread-safe LRU cache for duplicate queries.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return RecommendationIntent()

        # 1. Check cache
        cached_intent = self._cache.get(cleaned_query)
        if cached_intent is not None:
            logger.info(f"[Intent Cache] HIT for query: '{cleaned_query}'")
            return cached_intent

        query_cleaned = cleaned_query.lower()

        # 2. Add Local Intent Router Before LLM: Exact Match
        matched_title = None
        for title in self.movie_titles:
            if title.lower() == query_cleaned:
                matched_title = title
                break

        # 3. Fuzzy Match Check (only for queries that are likely movie names)
        if not matched_title and self.movie_titles:
            # difflib is fast enough for n=1 with high cutoff
            matches = difflib.get_close_matches(cleaned_query, self.movie_titles, n=1, cutoff=0.88)
            if matches:
                matched_title = matches[0]

        if matched_title:
            logger.info(f"[Local Intent Router] Bypass LLM. Query '{cleaned_query}' matched movie title: '{matched_title}'")
            local_intent = RecommendationIntent(
                intent="movie_lookup",
                ranking_mode="default",
                similar_movies=[matched_title]
            )
            self._cache.set(cleaned_query, local_intent)
            return local_intent

        # 4. Check if query is extremely simple (no LLM call required)
        words = re.findall(r'[a-zA-Z0-9\-]+', query_cleaned)
        if len(words) <= 3:
            genre_words = {"action", "adventure", "animation", "comedy", "crime", "documentary",
                           "drama", "family", "fantasy", "history", "horror", "music",
                           "mystery", "romance", "sci-fi", "science-fiction", "science fiction",
                           "thriller", "war", "western"}
            allowed_fillers = {"movie", "movies", "film", "films", "show", "shows", "best", "top", "good", "nice", "a", "the", "like"}
            if all(w in genre_words or w in allowed_fillers for w in words):
                logger.info(f"[Local Intent Router] Simple query detected: '{cleaned_query}'. Using local heuristic parser.")
                simple_intent = self._extract_intent_heuristically(cleaned_query)
                self._cache.set(cleaned_query, simple_intent)
                return simple_intent

        # 5. Call LLM for extraction
        logger.info(f"[Intent Cache] MISS for query: '{cleaned_query}'. Calling {self.provider} API...")
        
        system_prompt = (
            "You extract movie intent.\n"
            "Return ONLY valid JSON.\n"
            "No markdown.\n"
            "No explanation.\n"
            "No additional text.\n"
            "Return exactly this schema:\n"
            "{\n"
            '  "intent":"",\n'
            '  "movie_name":"",\n'
            '  "genres":[],\n'
            '  "similar_movies":[],\n'
            '  "keywords":[],\n'
            '  "ranking_mode":""\n'
            "}"
        )
        user_prompt = cleaned_query
        
        t_start = time.perf_counter()
        
        try:
            content = await self.extract_llm(system_prompt, user_prompt)
            latency_ms = (time.perf_counter() - t_start) * 1000
            
            # Estimate token counts (roughly 4 chars per token)
            est_input_tokens = round((len(system_prompt) + len(user_prompt)) / 4)
            est_output_tokens = round(len(content) / 4)
            logger.info(
                f"[LLM Intent] Success. Provider: {self.provider}, Model: {self.model}, "
                f"Prompt Tokens (est): {est_input_tokens}, Completion Tokens (est): {est_output_tokens}, "
                f"Latency: {latency_ms:.2f} ms"
            )
            
            # 6. Parse and validate JSON
            intent = self._parse_and_validate_json(content)
            if intent:
                self._cache.set(cleaned_query, intent)
                return intent
            else:
                logger.warning("[LLM Intent] JSON validation failed. Attempting one lightweight repair...")
                repaired_content = self._attempt_json_repair(content)
                intent = self._parse_and_validate_json(repaired_content)
                if intent:
                    logger.info("[LLM Intent] JSON repair succeeded.")
                    self._cache.set(cleaned_query, intent)
                    return intent
                else:
                    logger.error("[LLM Intent] JSON repair failed. Falling back to local heuristic parser.")
                    heuristic_intent = self._extract_intent_heuristically(cleaned_query)
                    self._cache.set(cleaned_query, heuristic_intent)
                    return heuristic_intent
                    
        except Exception as e:
            # Handle error logging: Timeout, Network, 429, 500, json_validate_failed, max tokens, etc.
            err_type = type(e).__name__
            logger.error(f"[LLM Intent] API error: {err_type} - {e}. Falling back to local heuristic parser.")
            heuristic_intent = self._extract_intent_heuristically(cleaned_query)
            self._cache.set(cleaned_query, heuristic_intent)
            return heuristic_intent

    def _parse_and_validate_json(self, content: str) -> Optional[RecommendationIntent]:
        """Parses and validates LLM completion JSON content against the schema."""
        try:
            cleaned = content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            data = json.loads(cleaned)
            
            # Map simplified schema to full RecommendationIntent model
            ranking_mode = (data.get("ranking_mode") or "").strip().lower()
            if ranking_mode not in ["best", "similar", "mood", "discover", "classic", "underrated", "recent"]:
                if "similar" in ranking_mode:
                    ranking_mode = "similar"
                elif "best" in ranking_mode:
                    ranking_mode = "best"
                else:
                    ranking_mode = "default"

            mapped_data = {
                "intent": data.get("intent") or "recommendation",
                "ranking_mode": ranking_mode,
                "genres": data.get("genres") or [],
                "similar_movies": data.get("similar_movies") or [],
                "keywords": data.get("keywords") or [],
            }
            
            # Merge movie_name to similar_movies
            movie_name = data.get("movie_name")
            if movie_name and movie_name not in mapped_data["similar_movies"]:
                mapped_data["similar_movies"] = [movie_name] + mapped_data["similar_movies"]
                
            return RecommendationIntent(**mapped_data)
        except Exception as e:
            logger.debug(f"JSON parsing/validation failed: {e}")
            return None

    def _attempt_json_repair(self, content: str) -> str:
        """Lightweight JSON repair for common formatting errors (unclosed brackets/braces/quotes)."""
        repaired = content.strip()
        
        # 1. Strip markdown wrapper
        if repaired.startswith("```json"):
            repaired = repaired[7:]
        if repaired.startswith("```"):
            repaired = repaired[3:]
        if repaired.endswith("```"):
            repaired = repaired[:-3]
        repaired = repaired.strip()
        
        # 2. Basic bracket/brace balancing
        open_braces = repaired.count("{")
        close_braces = repaired.count("}")
        if open_braces > close_braces:
            repaired += "}" * (open_braces - close_braces)
            
        open_brackets = repaired.count("[")
        close_brackets = repaired.count("]")
        if open_brackets > close_brackets:
            repaired += "]" * (open_brackets - close_brackets)
            
        # 3. Remove trailing comma before closing brace/bracket
        repaired = re.sub(r',\s*([\]}])', r'\1', repaired)
        
        return repaired

    def _extract_intent_heuristically(self, query: str) -> RecommendationIntent:
        """Fallback local heuristic intent parser using regex matching."""
        logger.info(f"[Heuristic Parser] Running heuristic fallback extraction for query: '{query}'")
        q = query.lower()
        
        # 1. Determine Intent
        intent = "recommendation"
        if re.search(r'\b(tell me about|details for|lookup|show details|info for|trailer|details|about)\b', q):
            intent = "movie_lookup"
        elif re.search(r'\b(search|find|list|show me|movies directed by|movies from)\b', q):
            if not re.search(r'\b(recommend|like|similar to|suggest|recommendation)\b', q):
                intent = "general_search"
                
        # 2. Determine Ranking Mode
        ranking_mode = "default"
        if re.search(r'\b(best|top|masterpiece|award|oscar|highly rated|acclaimed)\b', q):
            ranking_mode = "best"
        elif re.search(r'\b(like|similar to|resemble|type of)\b', q):
            ranking_mode = "similar_movie"
        elif re.search(r'\b(dark|funny|feel-good|scary|romantic|intense|thrilling|spooky|creepy|hilarious|heartwarming|suspenseful|comedy)\b', q):
            ranking_mode = "mood"

        # 3. Extract Genres
        canonical_genres = {
            "action": "Action", "adventure": "Adventure", "animation": "Animation",
            "comedy": "Comedy", "crime": "Crime", "documentary": "Documentary",
            "drama": "Drama", "family": "Family", "fantasy": "Fantasy",
            "history": "History", "horror": "Horror", "music": "Music",
            "mystery": "Mystery", "romance": "Romance", "science fiction": "Sci-Fi",
            "sci-fi": "Sci-Fi", "tv movie": "TV Movie", "thriller": "Thriller",
            "war": "War", "western": "Western"
        }
        genres = []
        for gkey, gval in canonical_genres.items():
            if re.search(r'\b' + re.escape(gkey) + r'\b', q):
                if gval not in genres:
                    genres.append(gval)

        # 4. Extract Moods and Themes
        mood_keywords = ["dark", "funny", "feel-good", "scary", "romantic", "intense", "thrilling", "spooky", "creepy", "hilarious", "heartwarming", "suspenseful"]
        moods = []
        for mood in mood_keywords:
            if re.search(r'\b' + re.escape(mood) + r'\b', q):
                moods.append(mood)
                
        theme_keywords = ["mind-bending", "heist", "time travel", "space travel", "revenge", "dystopian", "cyberpunk", "magic"]
        themes = []
        for theme in theme_keywords:
            if re.search(r'\b' + re.escape(theme) + r'\b', q):
                themes.append(theme)

        # 5. Extract Years
        release_year = None
        year_range = None
        
        range_match = re.search(r'\b(?:between\s+)?(19\d{2}|20\d{2})\s*(?:and|to|-)\s*(19\d{2}|20\d{2})\b', q)
        if range_match:
            y1, y2 = int(range_match.group(1)), int(range_match.group(2))
            year_range = YearRange(start=y1, end=y2)
        else:
            after_match = re.search(r'\b(?:after|since)\s+(19\d{2}|20\d{2})\b', q)
            if after_match:
                year_range = YearRange(start=int(after_match.group(1)))
            else:
                before_match = re.search(r'\b(?:before|until)\s+(19\d{2}|20\d{2})\b', q)
                if before_match:
                    year_range = YearRange(end=int(before_match.group(1)))
                else:
                    single_match = re.search(r'\b(19\d{2}|20\d{2})\b', q)
                    if single_match:
                        yr = int(single_match.group(1))
                        release_year = yr
                        year_range = YearRange(start=yr, end=yr)

        # 6. Extract Exclusions
        exclusions = []
        avoid_genres = []
        excl_match = re.search(r'\b(?:without|no|except|excluding|avoid|minus)\s+([a-zA-Z0-9\s,-]+)', q)
        if excl_match:
            raw_excl = excl_match.group(1).strip()
            parts = [p.strip() for p in re.split(r',|\band\b', raw_excl) if p.strip()]
            for p in parts:
                exclusions.append(p)
                if p in canonical_genres:
                    avoid_genres.append(canonical_genres[p])

        # 7. Extract Similar Movies (Reference Movies)
        similar_movies = []
        sim_match = re.search(r'\b(?:like|similar to|similar)\s+([A-Z][a-zA-Z0-9\s:\-\.]+)', query)
        if sim_match:
            title_candidate = sim_match.group(1).strip()
            title_candidate = re.split(r'\b(without|no|except|excluding|avoid|minus|starring|directed by|in|from|between)\b', title_candidate, flags=re.I)[0].strip()
            if title_candidate:
                similar_movies.append(title_candidate)
        elif intent == "movie_lookup":
            title_match = re.search(r'\b(?:about|details for|lookup|show details|info for|details)\s+([A-Z][a-zA-Z0-9\s:\-\.]+)', query)
            if title_match:
                title_candidate = title_match.group(1).strip()
                title_candidate = re.split(r'\b(without|no|except|excluding|avoid|minus|starring|directed by|in|from|between)\b', title_candidate, flags=re.I)[0].strip()
                if title_candidate:
                    similar_movies.append(title_candidate)
            else:
                clean_t = re.sub(r'\b(tell me about|details for|lookup|show details|info for|details|info|about)\b', '', query, flags=re.I).strip()
                if clean_t:
                    similar_movies.append(clean_t)

        # 8. Extract Crew/Actors
        preferred_actors = []
        actor_match = re.search(r'\b(?:starring|with|actors|actor)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)', query)
        if actor_match:
            preferred_actors.append(actor_match.group(1).strip())
            
        preferred_directors = []
        dir_match = re.search(r'\b(?:directed by|director)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)', query)
        if dir_match:
            preferred_directors.append(dir_match.group(1).strip())

        return RecommendationIntent(
            intent=intent,
            ranking_mode=ranking_mode,
            genres=genres,
            moods=moods,
            themes=themes,
            similar_movies=similar_movies,
            preferred_actors=preferred_actors,
            preferred_directors=preferred_directors,
            year_range=year_range,
            release_year=release_year,
            exclusions=exclusions,
            avoid_genres=avoid_genres
        )
