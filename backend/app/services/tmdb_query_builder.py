import asyncio
from typing import Dict, Any, List
from loguru import logger
from app.services.intent_extractor import RecommendationIntent
from app.services.tmdb_service import TMDbService

TMDB_GENRE_MAP = {
    "action": 28,
    "adventure": 12,
    "animation": 16,
    "comedy": 35,
    "crime": 80,
    "documentary": 99,
    "drama": 18,
    "family": 10751,
    "fantasy": 14,
    "history": 36,
    "horror": 27,
    "music": 10402,
    "mystery": 9648,
    "romance": 10749,
    "science fiction": 878,
    "sci-fi": 878,
    "tv movie": 10770,
    "thriller": 53,
    "war": 10752,
    "western": 37
}

TMDB_LANGUAGE_MAP = {
    "korean": "ko",
    "ko": "ko",
    "english": "en",
    "en": "en",
    "japanese": "ja",
    "ja": "ja",
    "french": "fr",
    "fr": "fr",
    "spanish": "es",
    "es": "es",
    "german": "de",
    "de": "de",
    "italian": "it",
    "it": "it",
    "chinese": "zh",
    "zh": "zh",
    "hindi": "hi",
    "hi": "hi",
}

class TMDbQueryBuilder:
    """Builder service to map RecommendationIntent to TMDb Discover API query parameters."""

    @staticmethod
    async def build_query(intent: RecommendationIntent, tmdb_service: TMDbService) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "include_adult": "false",
            "include_video": "false",
            "page": 1
        }

        # 1. Genres mapping
        genre_ids = []
        for g in intent.genres:
            g_lower = g.lower()
            if g_lower in TMDB_GENRE_MAP:
                genre_ids.append(str(TMDB_GENRE_MAP[g_lower]))
        if genre_ids:
            # Use OR operator (|) for broader candidate pool
            params["with_genres"] = "|".join(genre_ids)

        # 2. Excluded Genres mapping
        avoid_genre_ids = []
        for g in intent.avoid_genres:
            g_lower = g.lower()
            if g_lower in TMDB_GENRE_MAP:
                avoid_genre_ids.append(str(TMDB_GENRE_MAP[g_lower]))
        if avoid_genre_ids:
            # Excluded genres should remain AND-ed (comma-separated) to exclude all of them
            params["without_genres"] = ",".join(avoid_genre_ids)

        # 3. Language mapping
        if intent.language:
            lang_lower = intent.language.lower().strip()
            if lang_lower in TMDB_LANGUAGE_MAP:
                params["with_original_language"] = TMDB_LANGUAGE_MAP[lang_lower]
            elif len(lang_lower) == 2 and lang_lower in set(TMDB_LANGUAGE_MAP.values()):
                params["with_original_language"] = lang_lower

        # 4. Release year/range constraints
        if intent.release_year:
            params["primary_release_year"] = intent.release_year
        elif intent.year_range:
            if intent.year_range.start:
                params["primary_release_date.gte"] = f"{intent.year_range.start}-01-01"
            if intent.year_range.end:
                params["primary_release_date.lte"] = f"{intent.year_range.end}-12-31"

        # 5. Runtime constraint
        if intent.runtime:
            params["with_runtime.lte"] = intent.runtime

        # 6. Keywords/Themes/Moods mapping (Async resolution)
        all_keywords = list(set(intent.keywords + intent.themes + intent.moods))
        if all_keywords:
            tasks = [tmdb_service.resolve_keyword_id(kw) for kw in all_keywords]
            kw_ids = await asyncio.gather(*tasks)
            valid_kw_ids = [str(kid) for kid in kw_ids if kid is not None]
            if valid_kw_ids:
                # Use OR operator (|) for keywords to get a broader candidate pool
                params["with_keywords"] = "|".join(valid_kw_ids)

        # 7. Preferred Actors mapping (Async resolution)
        if intent.preferred_actors:
            tasks = [tmdb_service.resolve_person_id(act) for act in intent.preferred_actors]
            actor_ids = await asyncio.gather(*tasks)
            valid_actor_ids = [str(aid) for aid in actor_ids if aid is not None]
            if valid_actor_ids:
                params["with_cast"] = "|".join(valid_actor_ids)

        # 8. Preferred Directors mapping (Async resolution)
        if intent.preferred_directors:
            tasks = [tmdb_service.resolve_person_id(dr) for dr in intent.preferred_directors]
            dir_ids = await asyncio.gather(*tasks)
            valid_dir_ids = [str(did) for did in dir_ids if did is not None]
            if valid_dir_ids:
                params["with_crew"] = "|".join(valid_dir_ids)

        # 9. Ranking mode mapping (sorting, vote/rating quality guidelines)
        mode = getattr(intent, "ranking_mode", "default")
        if mode == "best":
            params["sort_by"] = "popularity.desc"
            params["vote_count.gte"] = 100
            params["vote_average.gte"] = 6.5
        elif mode == "similar_movie":
            params["sort_by"] = "popularity.desc"
            params["vote_count.gte"] = 50
        elif mode == "mood":
            params["sort_by"] = "popularity.desc"
        else: # default
            params["sort_by"] = "popularity.desc"
            params["vote_count.gte"] = 50

        return params
