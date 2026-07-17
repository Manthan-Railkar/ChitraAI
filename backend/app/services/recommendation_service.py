import asyncio
from typing import Any, List, Dict, Optional, Tuple
from loguru import logger
import polars as pl

from app.services.local_retrieval import LocalRetrievalEngine
from app.services.intent_extractor import IntentExtractor, RecommendationIntent
from app.services.tmdb_service import TMDbService
from app.services.enrichment_helper import enrich_movie_with_tmdb


THEME_EXPANSION_MAP = {
    "superhero": {
        "genres": ["Action", "Fantasy"],
        "keywords": ["superhero", "super power", "comic book", "masked superhero", "marvel cinematic universe", "dc extended universe"]
    },
    "mind-bending": {
        "genres": ["Mystery", "Thriller"],
        "keywords": ["mind-bending", "twist ending", "unreliable narrator", "reality distortion", "hallucination", "psychological"]
    },
    "time travel": {
        "genres": ["Science Fiction"],
        "keywords": ["time travel", "time loop", "wormhole", "temporal loop"]
    },
    "space": {
        "genres": ["Science Fiction", "Adventure"],
        "keywords": ["outer space", "space travel", "astronaut", "spaceship", "alien", "galaxy"]
    },
    "prison escape": {
        "genres": ["Thriller", "Action", "Drama"],
        "keywords": ["prison escape", "jailbreak", "prison", "escape"]
    },
    "heist": {
        "genres": ["Crime", "Thriller"],
        "keywords": ["heist", "bank robbery", "caper", "robbery", "thief"]
    },
    "serial killer": {
        "genres": ["Crime", "Thriller", "Mystery"],
        "keywords": ["serial killer", "psychopathic killer", "investigation", "murder mystery"]
    },
    "artificial intelligence": {
        "genres": ["Science Fiction"],
        "keywords": ["artificial intelligence", "android", "cyborg", "robot", "sentient computer"]
    },
    "cyberpunk": {
        "genres": ["Science Fiction"],
        "keywords": ["cyberpunk", "dystopia", "futuristic", "megacorporation", "neon"]
    },
    "zombie": {
        "genres": ["Horror", "Action"],
        "keywords": ["zombie", "undead", "apocalypse", "zombie apocalypse"]
    },
    "spy": {
        "genres": ["Action", "Thriller"],
        "keywords": ["spy", "espionage", "secret agent", "covert operation", "cia", "mi6"]
    },
    "coming of age": {
        "genres": ["Drama", "Comedy"],
        "keywords": ["coming of age", "teenage life", "adolescence", "growing up"]
    },
    "courtroom": {
        "genres": ["Drama"],
        "keywords": ["courtroom", "legal drama", "trial", "lawyer", "justice system"]
    },
    "sports": {
        "genres": ["Drama"],
        "keywords": ["sports", "athlete", "coach", "boxing", "football", "baseball", "tournament"]
    },
    "war": {
        "genres": ["War", "History"],
        "keywords": ["war", "soldier", "military", "battle", "combat"]
    },
    "historical": {
        "genres": ["History", "Drama"],
        "keywords": ["historical", "period piece", "biography", "true story"]
    },
    "political": {
        "genres": ["Drama", "Thriller"],
        "keywords": ["politics", "political corruption", "government", "conspiracy"]
    },
    "musical": {
        "genres": ["Music", "Romance"],
        "keywords": ["musical", "singing", "dancing"]
    },
    "adventure": {
        "genres": ["Adventure"],
        "keywords": ["adventure", "quest", "journey", "exploration"]
    },
    "psychological": {
        "genres": ["Mystery", "Thriller", "Drama"],
        "keywords": ["psychological", "mental illness", "identity crisis", "paranoia"]
    },
    "survival": {
        "genres": ["Thriller", "Adventure", "Drama"],
        "keywords": ["survival", "stranded", "wilderness survival", "hostile environment"]
    },
    "noir": {
        "genres": ["Crime", "Mystery", "Thriller"],
        "keywords": ["neo-noir", "film noir", "detective", "femme fatale", "cynical"]
    }
}


class RecommendationService:
    """
    Local recommendation engine for ChitraAI.
    Combines OpenAI intent extraction, structured filtering, local semantic similarity,
    and weighted scoring to retrieve the best candidate recommendations.
    """
    def __init__(self, local_retrieval_engine: LocalRetrievalEngine, intent_extractor: IntentExtractor, tmdb_service: Optional[TMDbService] = None) -> None:
        self.local_retrieval_engine = local_retrieval_engine
        self.intent_extractor = intent_extractor
        self.tmdb_service = tmdb_service or TMDbService()
        logger.info("RecommendationService initialized with local retrieval engine and OpenAI intent extractor.")

    def expand_abstract_themes(self, intent: RecommendationIntent, query: str) -> None:
        """
        Dynamically expands abstract query concepts into TMDb genres and keywords.
        """
        # Ensure initial genres are normalized
        intent.genres = RecommendationIntent._normalize_genre_list(intent.genres)
        intent.explicit_genres = list(intent.genres)
        query_lower = query.lower()
        
        # Collect all intent metadata elements to match against
        intent_elements = []
        for attr in ["genres", "keywords", "themes", "moods"]:
            val = getattr(intent, attr, [])
            if val:
                intent_elements.extend([item.lower().strip() for item in val])
        if getattr(intent, "complexity", None):
            intent_elements.append(intent.complexity.lower())
        if getattr(intent, "pacing", None):
            intent_elements.append(intent.pacing.lower())
            
        for theme_key, mapping in THEME_EXPANSION_MAP.items():
            # Check if theme key exists as a word/phrase in query or intent components
            is_matched = (theme_key in query_lower) or any(theme_key in elem for elem in intent_elements)
            if not is_matched and theme_key == "mind-bending" and "mind bending" in query_lower:
                is_matched = True
            if not is_matched and theme_key == "coming of age" and "coming-of-age" in query_lower:
                is_matched = True
            if not is_matched and theme_key == "prison escape" and "prison-escape" in query_lower:
                is_matched = True
            if not is_matched and theme_key == "serial killer" and "serial-killer" in query_lower:
                is_matched = True
            if not is_matched and theme_key == "courtroom" and "court room" in query_lower:
                is_matched = True
            if not is_matched and theme_key == "time travel" and "time-travel" in query_lower:
                is_matched = True

            if is_matched:
                # Add expanded genres (normalized)
                normalized_expanded_genres = RecommendationIntent._normalize_genre_list(mapping["genres"])
                for g in normalized_expanded_genres:
                    if g not in intent.genres:
                        intent.genres.append(g)
                # Add expanded keywords
                for kw in mapping["keywords"]:
                    if kw not in intent.keywords:
                        intent.keywords.append(kw)
                        
        logger.info(f"[Theme Expansion] Done. Genres: {intent.genres}, Keywords: {intent.keywords}")

    async def recommend_movies_from_query(self, query: str, limit: int = 50) -> Tuple[RecommendationIntent, List[Dict[str, Any]]]:
        """
        Extracts recommendation intent using OpenAI, runs structured filtering + semantic search,
        and returns the extracted intent along with the top candidate recommendations.
        """
        logger.info(f"Generating recommendations for query: '{query}' (limit={limit})")
        
        # 1. Extract structured intent using OpenAI
        intent = await self.intent_extractor.extract_intent(query)
        self.expand_abstract_themes(intent, query)

        # Validate person entities before routing (BUG #2, BUG #3, BUG #6)
        if getattr(intent, "strict_person_filter", False) and (intent.preferred_directors or intent.preferred_actors):
            target_directors = [d for d in intent.preferred_directors if d]
            target_actors = [a for a in intent.preferred_actors if a]
            
            dir_tasks = [self.tmdb_service.resolve_person_id(d) for d in target_directors]
            act_tasks = [self.tmdb_service.resolve_person_id(a) for a in target_actors]
            
            resolved_dirs = await asyncio.gather(*dir_tasks) if dir_tasks else []
            resolved_acts = await asyncio.gather(*act_tasks) if act_tasks else []
            
            valid_directors = [d for d, pid in zip(target_directors, resolved_dirs) if pid]
            valid_actors = [a for a, pid in zip(target_actors, resolved_acts) if pid]
            
            if not valid_directors and not valid_actors:
                logger.info(f"[Audit Log] No valid person entities found on TMDb. Disabling strict person filter. (Original: directors={target_directors}, actors={target_actors})")
                intent.strict_person_filter = False
                intent.preferred_directors = []
                intent.preferred_actors = []
            else:
                intent.preferred_directors = valid_directors
                intent.preferred_actors = valid_actors
                logger.info(f"[Audit Log] Validated strict person entities. Keep directors={valid_directors}, actors={valid_actors}")
        
        
        # 1b. Check if the intent is movie_lookup
        if getattr(intent, "intent", "recommendation") == "movie_lookup":
            search_title = intent.similar_movies[0] if intent.similar_movies else query
            search_year = getattr(intent, "release_year", None)
            if not search_year and intent.year_range:
                search_year = intent.year_range.start or intent.year_range.end
                
            logger.info(f"[Movie Lookup Bypass] Query '{query}' classified as movie_lookup. Searching TMDb for title='{search_title}', year={search_year}")
            
            tmdb_id = None
            movie_details = None
            if self.tmdb_service is not None:
                try:
                    tmdb_id = await self.tmdb_service.search_movie_by_title(search_title, year=search_year)
                    if tmdb_id:
                        movie_details = await self.tmdb_service.fetch_movie_details(tmdb_id)
                except Exception as e:
                    logger.error(f"[Movie Lookup Bypass] Failed to search or fetch from TMDb: {e}")
            
            if movie_details:
                movie_dict = {
                    "tmdb_id": tmdb_id,
                    "title": movie_details.get("title", search_title),
                    "original_title": movie_details.get("original_title"),
                    "overview": movie_details.get("overview"),
                    "genres": [g.get("name") for g in movie_details.get("genres", []) if g.get("name")],
                    "release_year": int(movie_details.get("release_date", "0000")[:4]) if movie_details.get("release_date") else None,
                    "rating_value": movie_details.get("vote_average"),
                    "vote_count": movie_details.get("vote_count"),
                    "popularity": movie_details.get("popularity"),
                    "poster_path": movie_details.get("poster_path"),
                    "backdrop_path": movie_details.get("backdrop_path"),
                    "runtime_minutes": movie_details.get("runtime"),
                    "recommendation_reason": "Exact match found in TMDb lookup.",
                    "retrieval_score": 1.0,
                    "boosted_semantic_score": 1.0,
                    "reranked_score": 1.0
                }
                
                credits = movie_details.get("credits", {})
                if credits:
                    directors = [m.get("name") for m in credits.get("crew", []) if m.get("job") == "Director"]
                    movie_dict["directors"] = directors
                    cast = [m.get("name") for m in credits.get("cast", [])[:10]]
                    movie_dict["cast"] = cast
                
                enriched_movie = await enrich_movie_with_tmdb(movie_dict, self.tmdb_service)
                return intent, [enriched_movie]
            
            # Local Database Fallback
            logger.info(f"[Movie Lookup Fallback] Movie '{search_title}' not found via TMDb search. Falling back to local database search...")
            if self.local_retrieval_engine.movies_df is None:
                self.local_retrieval_engine.initialize()
            
            df = self.local_retrieval_engine.movies_df
            if df is None:
                logger.warning(f"[Movie Lookup Fallback] Local database not initialized.")
                return intent, []
                
            # First try exact match
            matched_df = df.filter(
                pl.col("title").str.to_lowercase() == search_title.lower()
            )
            
            # If no exact match, try substring match but sort by difflib similarity ratio to the search query
            if matched_df.height == 0:
                substr_matched_df = df.filter(
                    pl.col("title").str.to_lowercase().str.contains(search_title.lower())
                )
                if substr_matched_df.height > 0:
                    import difflib
                    records = substr_matched_df.to_dicts()
                    # Rank records by string similarity to search_title
                    records.sort(
                        key=lambda x: difflib.SequenceMatcher(None, (x.get("title") or "").lower(), search_title.lower()).ratio(),
                        reverse=True
                    )
                    best_match = records[0]
                else:
                    best_match = None
            else:
                best_match = matched_df.sort("popularity", descending=True).to_dicts()[0]

            if best_match:
                best_match["recommendation_reason"] = "Exact match found in local database lookup."
                best_match["retrieval_score"] = 1.0
                best_match["boosted_semantic_score"] = 1.0
                best_match["reranked_score"] = 1.0
                
                enriched_movie = await enrich_movie_with_tmdb(best_match, self.tmdb_service)
                return intent, [enriched_movie]
                
            logger.warning(f"[Movie Lookup Fallback] Movie '{search_title}' not found in TMDb or local database.")
            return intent, []

        # 2. Retrieve candidates from TMDb Discover if enabled, otherwise fallback to local engine
        from app.core.config import settings
        if settings.USE_TMDB_RETRIEVAL and self.tmdb_service and self.tmdb_service.api_key:
            try:
                import time as _time
                t_pipeline_start = _time.perf_counter()
                mode = getattr(intent, "ranking_mode", "default").strip().lower()
                logger.info(f"[Intelligent Retrieval] Selected retrieval strategy based on mode: '{mode}'")

                # --- DETERMINE PIPELINE PATHWAY ---
                pipeline_pathway = "funnel"  # default
                reference_vector = None

                # Pathway B: Similarity engine (when similar_movies is populated and mode is similar)
                has_similar = bool(intent.similar_movies)
                is_similar_mode = mode in ("similar", "similar_movie")
                if has_similar and is_similar_mode:
                    pipeline_pathway = "similarity"

                # Pathway C: Strict person filter
                if getattr(intent, "strict_person_filter", False) and (intent.preferred_directors or intent.preferred_actors):
                    pipeline_pathway = "strict_person"

                logger.info(f"[Pipeline] Selected pathway: '{pipeline_pathway}'")

                # --- PATHWAY B: SIMILARITY ENGINE ---
                if pipeline_pathway == "similarity":
                    logger.info(f"[Similarity Engine] Routing through similarity pathway for: {intent.similar_movies}")

                    # Resolve reference movie IDs
                    sim_resolve_tasks = [self.tmdb_service.search_movie_by_title(sim) for sim in intent.similar_movies if sim]
                    sim_resolved_ids = await asyncio.gather(*sim_resolve_tasks)

                    # Fetch similar + recommended + details for each resolved reference
                    sim_fetch_tasks = []
                    sim_fetch_labels = []
                    for sim_title, sim_id in zip(intent.similar_movies, sim_resolved_ids):
                        if sim_id:
                            sim_fetch_tasks.append(self.tmdb_service.fetch_similar(sim_id))
                            sim_fetch_labels.append(f"similar_to_{sim_title}")
                            sim_fetch_tasks.append(self.tmdb_service.fetch_recommendations(sim_id))
                            sim_fetch_labels.append(f"recs_for_{sim_title}")
                            sim_fetch_tasks.append(self.tmdb_service.fetch_movie_details(sim_id))
                            sim_fetch_labels.append(f"details_of_{sim_title}")

                    sim_responses = await asyncio.gather(*sim_fetch_tasks) if sim_fetch_tasks else []

                    # Enrich intent context from reference movie details (from the gathered responses)
                    for lbl, resp in zip(sim_fetch_labels, sim_responses):
                        if "details_of_" in lbl and resp:
                            ref_genres = [g.get("name") for g in resp.get("genres", []) if g.get("name")]
                            ref_genres_norm = RecommendationIntent._normalize_genre_list(ref_genres)
                            ref_kws = [k.get("name") for k in resp.get("keywords", {}).get("keywords", []) if k.get("name")]
                            for g in ref_genres_norm:
                                if g not in intent.genres:
                                    intent.genres.append(g)
                            for k in ref_kws[:10]:
                                if k not in intent.keywords:
                                    intent.keywords.append(k)

                    # Resolve reference movie vector (from local cache if available, or encode metadata details on the fly)
                    if sim_resolved_ids:
                        for ref_id in sim_resolved_ids:
                            if ref_id and ref_id in self.local_retrieval_engine.tmdb_id_to_idx and self.local_retrieval_engine.embeddings_matrix is not None:
                                idx = self.local_retrieval_engine.tmdb_id_to_idx[ref_id]
                                reference_vector = self.local_retrieval_engine.embeddings_matrix[idx]
                                logger.info(f"[Similarity Pathway] Using precomputed embedding for reference movie ID: {ref_id}")
                                break

                        if reference_vector is None:
                            from app.services.local_retrieval import build_embedding_document
                            for lbl, resp in zip(sim_fetch_labels, sim_responses):
                                if "details_of_" in lbl and resp:
                                    ref_doc = build_embedding_document(
                                        title=resp.get("title"),
                                        tagline=resp.get("tagline"),
                                        overview=resp.get("overview"),
                                        genres=[g.get("name") for g in resp.get("genres", []) if g.get("name")],
                                        keywords=[k.get("name") for k in resp.get("keywords", {}).get("keywords", []) if k.get("name")],
                                        cast=[m.get("name") for m in resp.get("credits", {}).get("cast", [])[:10]],
                                        directors=[m.get("name") for m in resp.get("credits", {}).get("crew", []) if m.get("job") == "Director"]
                                    )
                                    reference_vector = self.local_retrieval_engine.embedding_service.encode_single(ref_doc, normalize=True)
                                    logger.info(f"[Similarity Pathway] Encoded fetched details for reference movie: '{resp.get('title')}'")
                                    break

                    unique_stubs = {}
                    for lbl, resp in zip(sim_fetch_labels, sim_responses):
                        if not resp:
                            continue
                        if "details_of_" in lbl:
                            tmdb_id = resp.get("id")
                            if tmdb_id and tmdb_id not in unique_stubs:
                                unique_stubs[tmdb_id] = resp
                        else:
                            results_list = resp.get("results", []) if isinstance(resp, dict) else []
                            for m in results_list:
                                tmdb_id = m.get("id")
                                if tmdb_id and tmdb_id not in unique_stubs:
                                    unique_stubs[tmdb_id] = m

                    logger.info(f"[Similarity Engine] Total unique candidate stubs: {len(unique_stubs)}")
                    candidates = await self._enrich_stubs_to_candidates(unique_stubs, limit=50)
                    all_labels = sim_fetch_labels
                    source_movies_log = {lbl: "similarity_engine" for lbl in sim_fetch_labels}

                # --- PATHWAY C: STRICT PERSON FILTER ---
                elif pipeline_pathway == "strict_person":
                    target_persons = intent.preferred_directors or intent.preferred_actors
                    is_director_query = bool(intent.preferred_directors)
                    logger.info(f"[Strict Person] Fetching filmography for: {target_persons} (director={is_director_query})")

                    # Resolve person IDs
                    person_resolve_tasks = [self.tmdb_service.resolve_person_id(p) for p in target_persons if p]
                    person_ids = await asyncio.gather(*person_resolve_tasks)

                    # Fetch filmographies
                    filmography_tasks = [self.tmdb_service.fetch_person_movie_credits(pid) for pid in person_ids if pid]
                    filmography_responses = await asyncio.gather(*filmography_tasks)

                    unique_stubs = {}
                    for resp in filmography_responses:
                        if not resp:
                            continue
                        if is_director_query:
                            for m in resp.get("crew", []):
                                if m.get("job") == "Director":
                                    tmdb_id = m.get("id")
                                    if tmdb_id and tmdb_id not in unique_stubs:
                                        unique_stubs[tmdb_id] = m
                        else:
                            cast_sorted = sorted(resp.get("cast", []), key=lambda x: x.get("popularity") or 0.0, reverse=True)
                            for m in cast_sorted[:60]:
                                tmdb_id = m.get("id")
                                if tmdb_id and tmdb_id not in unique_stubs:
                                    unique_stubs[tmdb_id] = m

                    logger.info(f"[Strict Person] Total filmography stubs: {len(unique_stubs)}")
                    candidates = await self._enrich_stubs_to_candidates(unique_stubs, limit=50)
                    all_labels = [f"filmography_{p}" for p in target_persons]
                    source_movies_log = {f"filmography_{p}": f"{len(unique_stubs)} movies" for p in target_persons}

                # --- PATHWAY D: MULTI-STAGE FUNNEL (default) ---
                else:
                    from app.services.tmdb_query_builder import TMDbQueryBuilder
                    discover_params = await TMDbQueryBuilder.build_query(intent, self.tmdb_service)
                    logger.info(f"[TMDb Retrieval] Primary Discover parameters: {discover_params}")

                    # Resolve similar movie reference names for intent context enrichment
                    ref_titles = intent.similar_movies or ([intent.movie_name] if getattr(intent, "movie_name", None) else [])
                    if ref_titles:
                        first_ref = ref_titles[0]
                        logger.info(f"[Context Boost] Resolving reference movie '{first_ref}' to enrich intent context...")
                        try:
                            ref_id = await self.tmdb_service.search_movie_by_title(first_ref)
                            if ref_id:
                                ref_details = await self.tmdb_service.fetch_movie_details(ref_id)
                                if ref_details:
                                    ref_genres = [g.get("name") for g in ref_details.get("genres", []) if g.get("name")]
                                    ref_genres_norm = RecommendationIntent._normalize_genre_list(ref_genres)
                                    ref_kws = [k.get("name") for k in ref_details.get("keywords", {}).get("keywords", []) if k.get("name")]
                                    ref_directors = [c.get("name") for c in ref_details.get("credits", {}).get("crew", []) if c.get("job") == "Director" and c.get("name")]
                                    ref_collection_name = ref_details.get("belongs_to_collection", {}).get("name") if ref_details.get("belongs_to_collection") else None

                                    for g in ref_genres_norm:
                                        if g not in intent.genres:
                                            intent.genres.append(g)
                                    for k in ref_kws[:10]:
                                        if k not in intent.keywords:
                                            intent.keywords.append(k)
                                    for d in ref_directors:
                                        if d not in intent.preferred_directors:
                                            intent.preferred_directors.append(d)
                                    if ref_collection_name:
                                        intent.ref_collections = [ref_collection_name]
                                    logger.info(f"[Context Boost] Successfully enriched intent from '{first_ref}' with genres={ref_genres}, directors={ref_directors}, collection='{ref_collection_name}'")
                        except Exception as e:
                            logger.error(f"[Context Boost] Failed to resolve reference movie details: {e}")

                    # 1. Resolve actors, directors, and similar movies concurrently
                    resolve_tasks = []
                    resolve_labels = []

                    for d in intent.preferred_directors:
                        if d:
                            resolve_tasks.append(self.tmdb_service.resolve_person_id(d))
                            resolve_labels.append(f"dir_{d}")
                    for a in intent.preferred_actors:
                        if a:
                            resolve_tasks.append(self.tmdb_service.resolve_person_id(a))
                            resolve_labels.append(f"act_{a}")
                    for sim in intent.similar_movies:
                        if sim:
                            resolve_tasks.append(self.tmdb_service.search_movie_by_title(sim))
                            resolve_labels.append(f"sim_{sim}")

                    resolved_ids = {}
                    if resolve_tasks:
                        logger.info(f"[Intelligent Retrieval] Resolving TMDb IDs for: {resolve_labels}...")
                        resolve_results = await asyncio.gather(*resolve_tasks)
                        for lbl, res in zip(resolve_labels, resolve_results):
                            resolved_ids[lbl] = res

                    # 2. Build composable retrieval coroutines
                    tasks_with_labels = []

                    # Baseline discover queries (pages 1-3)
                    tasks_with_labels.append(("discover_p1", self.tmdb_service.discover_movies(discover_params)))
                    tasks_with_labels.append(("discover_p2", self.tmdb_service.discover_movies({**discover_params, "page": 2})))
                    tasks_with_labels.append(("discover_p3", self.tmdb_service.discover_movies({**discover_params, "page": 3})))

                    # Company/Studio-specific discover queries if studio is specified
                    studio_ids = {
                        "ghibli": 10390, "pixar": 3, "marvel": 420, "lucasfilm": 1, "a24": 41077,
                        "warner": 174, "disney": 2, "universal": 33, "paramount": 4, "columbia": 5,
                    }
                    if intent.studio:
                        st_lower = intent.studio.lower()
                        matched_company_id = None
                        for key, val in studio_ids.items():
                            if key in st_lower:
                                matched_company_id = val
                                break
                        if matched_company_id:
                            tasks_with_labels.append((
                                f"studio_{intent.studio}",
                                self.tmdb_service.discover_movies({**discover_params, "with_companies": str(matched_company_id)})
                            ))

                    # Build filmography coroutines for resolved crew
                    person_filmography_tasks = []
                    person_labels = []
                    for d in intent.preferred_directors:
                        p_id = resolved_ids.get(f"dir_{d}")
                        if p_id:
                            person_filmography_tasks.append(self.tmdb_service.fetch_person_movie_credits(p_id))
                            person_labels.append(f"filmography_dir_{d}")
                    for a in intent.preferred_actors:
                        p_id = resolved_ids.get(f"act_{a}")
                        if p_id:
                            person_filmography_tasks.append(self.tmdb_service.fetch_person_movie_credits(p_id))
                            person_labels.append(f"filmography_act_{a}")

                    # Build similar and recommended movie coroutines
                    similar_tasks = []
                    similar_labels = []
                    for sim in intent.similar_movies:
                        m_id = resolved_ids.get(f"sim_{sim}")
                        if m_id:
                            similar_tasks.append(self.tmdb_service.fetch_similar(m_id))
                            similar_labels.append(f"similar_to_{sim}")
                            similar_tasks.append(self.tmdb_service.fetch_recommendations(m_id))
                            similar_labels.append(f"recs_for_{sim}")
                            similar_tasks.append(self.tmdb_service.fetch_movie_details(m_id))
                            similar_labels.append(f"details_of_{sim}")

                    # Build lists based on novelty and critical acclaim preferences
                    if intent.novelty_preference in ("recent", "new_release", "now_playing") or mode in ("recent", "recent_release"):
                        tasks_with_labels.append(("now_playing", self.tmdb_service.fetch_now_playing()))
                        tasks_with_labels.append(("upcoming", self.tmdb_service.fetch_upcoming()))
                    elif intent.novelty_preference == "trending" or mode in ("trending", "popular"):
                        tasks_with_labels.append(("trending", self.tmdb_service.fetch_trending()))
                        tasks_with_labels.append(("popular", self.tmdb_service.fetch_popular()))
                    elif intent.novelty_preference == "classic" or intent.critical_acclaim_preference == "high" or intent.awards_preference in ("oscar", "any") or mode == "best":
                        tasks_with_labels.append(("top_rated", self.tmdb_service.fetch_top_rated()))

                    # Execute all parallel retrieval tasks
                    all_tasks = [t[1] for t in tasks_with_labels] + person_filmography_tasks + similar_tasks
                    all_labels = [t[0] for t in tasks_with_labels] + person_labels + similar_labels
                    logger.info(f"[Intelligent Retrieval] Triggering parallel candidate queries for: {all_labels}")
                    
                    retrieval_responses = await asyncio.gather(*all_tasks)

                    # Merge, deduplicate, and identify franchise collections
                    unique_stubs = {}
                    collection_ids_to_fetch = set()
                    source_movies_log = {}

                    for lbl, resp in zip(all_labels, retrieval_responses):
                        if not resp:
                            continue
                        
                        if "filmography_dir_" in lbl:
                            crew = resp.get("crew", [])
                            source_movies_log[lbl] = [f"{m.get('title')} ({m.get('id')})" for m in crew[:5]]
                            for m in crew:
                                if m.get("job") == "Director":
                                    tmdb_id = m.get("id")
                                    if tmdb_id and tmdb_id not in unique_stubs:
                                        unique_stubs[tmdb_id] = m
                        elif "filmography_act_" in lbl:
                            cast = resp.get("cast", [])
                            cast_sorted = sorted(cast, key=lambda x: x.get("popularity") or 0.0, reverse=True)
                            source_movies_log[lbl] = [f"{m.get('title')} ({m.get('id')})" for m in cast_sorted[:5]]
                            for m in cast_sorted[:40]:
                                tmdb_id = m.get("id")
                                if tmdb_id and tmdb_id not in unique_stubs:
                                    unique_stubs[tmdb_id] = m
                        elif "details_of_" in lbl:
                            tmdb_id = resp.get("id")
                            if tmdb_id and tmdb_id not in unique_stubs:
                                unique_stubs[tmdb_id] = resp
                            coll = resp.get("belongs_to_collection")
                            if coll and coll.get("id"):
                                collection_ids_to_fetch.add(coll.get("id"))
                        else:
                            results_list = resp.get("results", []) if isinstance(resp, dict) else []
                            source_movies_log[lbl] = [f"{m.get('title')} ({m.get('id')})" for m in results_list[:5]]
                            for m in results_list:
                                tmdb_id = m.get("id")
                                if tmdb_id and tmdb_id not in unique_stubs:
                                    unique_stubs[tmdb_id] = m

                    # 3. Fetch collection details if any franchise collection was identified
                    if collection_ids_to_fetch:
                        logger.info(f"[Intelligent Retrieval] Fetching collection members for collection IDs: {collection_ids_to_fetch}")
                        col_tasks = [self.tmdb_service.fetch_collection_details(cid) for cid in collection_ids_to_fetch]
                        col_responses = await asyncio.gather(*col_tasks)
                        for idx, col_resp in enumerate(col_responses):
                            if col_resp and col_resp.get("parts"):
                                parts = col_resp.get("parts", [])
                                logger.info(f"[Intelligent Retrieval] Merged {len(parts)} members from collection: '{col_resp.get('name')}'")
                                for m in parts:
                                    tmdb_id = m.get("id")
                                    if tmdb_id and tmdb_id not in unique_stubs:
                                        unique_stubs[tmdb_id] = m

                    logger.info(f"[Intelligent Retrieval] Total unique candidate stubs merged: {len(unique_stubs)}")
                    candidates = await self._enrich_stubs_to_candidates(unique_stubs, limit=50)

                # ====== COMMON 3-STAGE FUNNEL (all pathways converge here) ======

                # Apply hard constraints BEFORE scoring and ranking
                from app.services.ranking_service import RankingService
                filtered_candidates = [c for c in candidates if RankingService.passes_hard_constraints(c, intent)]
                logger.info(f"[Stage 1] Candidates after hard constraints: {len(candidates)} -> {len(filtered_candidates)}")
                
                if not filtered_candidates:
                    logger.warning("[Stage 1] Hard constraints removed all candidates! Relaxing constraints to preserve query results.")
                    filtered_candidates = candidates
                
                candidates = filtered_candidates

                # --- STAGE 1: Broad Retrieval (take top 50 by heuristic) ---
                stage_1_candidates = sorted(
                    candidates,
                    key=lambda x: (x.get("popularity") or 0.0) * (x.get("rating_value") or 0.0),
                    reverse=True
                )[:50]
                logger.info(f"[Stage 1 - Broad Retrieval] {len(candidates)} candidates narrowed to {len(stage_1_candidates)} by heuristic stub score")

                def format_debug_candidate(movie: dict, final_score: float = 0.0) -> dict:
                    return {
                        "title": movie.get("title"),
                        "tmdb_id": movie.get("tmdb_id"),
                        "genres": movie.get("genres") or [],
                        "semantic_score": round(float(movie.get("semantic_score") or 0.0), 4),
                        "bm25_score": 0.0,
                        "metadata_match_score": 0.0,
                        "rating": movie.get("rating_value") or movie.get("rating"),
                        "vote_count": movie.get("vote_count"),
                        "popularity": movie.get("popularity"),
                        "final_score": round(final_score, 4)
                    }

                stage_1_log = [format_debug_candidate(m, final_score=(m.get("popularity") or 0.0) * (m.get("rating_value") or 0.0)) for m in stage_1_candidates]

                if stage_1_candidates:
                    # --- STAGE 2: Semantic Narrowing (50 -> 15) ---
                    if pipeline_pathway == "similarity" and reference_vector is not None:
                        query_vector = reference_vector
                    else:
                        query_doc = f"{query}. Genres: {', '.join(intent.genres)}. Themes: {', '.join(intent.themes)}. Moods: {', '.join(intent.moods)}. Keywords: {', '.join(intent.keywords)}."
                        query_vector = self.local_retrieval_engine.embedding_service.encode_single(query_doc, normalize=True)

                    def build_temp_document(movie: dict) -> str:
                        parts = [f"Title: {movie.get('title')}"]
                        if movie.get("genres"):
                            parts.append(f"Genres: {', '.join(movie.get('genres'))}")
                        if movie.get("directors"):
                            parts.append(f"Directed by: {', '.join(movie.get('directors'))}")
                        if movie.get("cast"):
                            parts.append(f"Starring: {', '.join(movie.get('cast'))}")
                        if movie.get("keywords"):
                            parts.append(f"Keywords: {', '.join(movie.get('keywords'))}")
                        if movie.get("overview"):
                            parts.append(movie.get("overview"))
                        return "\n".join(parts)

                    candidate_docs = [build_temp_document(m) for m in stage_1_candidates]
                    candidate_embs = self.local_retrieval_engine.embedding_service.encode_batch(candidate_docs, normalize=True)

                    import numpy as np
                    semantic_scores = np.dot(candidate_embs, query_vector)

                    # Pair candidates with their semantic scores and sort
                    scored_by_semantic = sorted(
                        zip(stage_1_candidates, semantic_scores.tolist()),
                        key=lambda x: x[1],
                        reverse=True
                    )

                    # Narrow to top 15 by semantic score
                    stage_2_pairs = scored_by_semantic[:15]
                    stage_2_candidates = [pair[0] for pair in stage_2_pairs]
                    stage_2_semantic_scores = [pair[1] for pair in stage_2_pairs]

                    logger.info(f"[Stage 2 - Semantic Narrowing] {len(stage_1_candidates)} candidates narrowed to {len(stage_2_candidates)} by semantic similarity")
                    stage_2_log = [format_debug_candidate(m, final_score=round(s, 4)) for m, s in stage_2_pairs]

                    # --- STAGE 3: Smart Ranking (15 -> final N) ---
                    from app.services.ranking_service import RankingService
                    scored_candidates = RankingService.rank_candidates(stage_2_candidates, intent, stage_2_semantic_scores)

                    diversified_candidates = RankingService.apply_diversity(scored_candidates)
                    top_candidates = diversified_candidates[:limit]

                    logger.info(f"[Stage 3 - Smart Ranking] {len(stage_2_candidates)} candidates ranked, diversified, and sliced to {len(top_candidates)}")
                    stage_3_log = [format_debug_candidate(m, final_score=m.get("retrieval_score") or 0.0) for m in top_candidates]

                    elapsed_ms = round((_time.perf_counter() - t_pipeline_start) * 1000, 2)

                    self.local_retrieval_engine.last_debug_report = {
                        "query": query,
                        "intent": intent.model_dump() if hasattr(intent, "model_dump") else str(intent),
                        "pipeline_pathway": pipeline_pathway,
                        "retrieval_path": "TMDb",
                        "candidate_sources": all_labels,
                        "movies_per_source": source_movies_log,
                        "stage_1_broad_retrieval": stage_1_log,
                        "stage_2_semantic_narrowing": stage_2_log,
                        "stage_3_final_ranking": stage_3_log,
                        "funnel_summary": {
                            "stage_1_count": len(stage_1_candidates),
                            "stage_2_count": len(stage_2_candidates),
                            "stage_3_count": len(top_candidates),
                            "execution_time_ms": elapsed_ms
                        },
                        "final_ranking_scores": {m.get("title"): round(float(m.get("retrieval_score") or 0.0), 4) for m in scored_candidates if m.get("title")},
                        "execution_time_ms": elapsed_ms
                    }

                    logger.info(f"[TMDb Pipeline] Success. Pathway={pipeline_pathway}, Candidates: {len(stage_1_candidates)}→{len(stage_2_candidates)}→{len(top_candidates)} in {elapsed_ms}ms")
                    from app.services.enrichment_helper import enrich_movie_list
                    enriched_candidates = await enrich_movie_list(top_candidates, self.tmdb_service)
                    return intent, enriched_candidates
                else:
                    logger.warning("[TMDb Retrieval] Discover returned 0 candidates. Falling back to local retrieval...")
            except Exception as e:
                logger.error(f"[TMDb Retrieval] Discover retrieval failed: {e}. Falling back to local retrieval...")

        # 3. Retrieve candidates locally (Fallback)
        results = await self.local_retrieval_engine.retrieve_candidates(
            original_query=query,
            intent=intent,
            limit=limit
        )
        from app.services.enrichment_helper import enrich_movie_list
        enriched_results = await enrich_movie_list(results, self.tmdb_service)
        return intent, enriched_results



    async def recommend_movies_from_understanding(
        self, understanding: Any, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Legacy compatibility method for existing tests or endpoints.
        """
        logger.warning("recommend_movies_from_understanding called (legacy method). Mapping to local engine.")
        # Map Gemini understanding to RecommendationIntent
        intent = RecommendationIntent(
            genres=getattr(understanding, "genres", []) or [],
            moods=[getattr(understanding, "mood", "")] if getattr(understanding, "mood", None) else [],
            themes=getattr(understanding, "themes", []) or [],
            similar_movies=getattr(understanding, "reference_movies", []) or [],
            preferred_actors=getattr(understanding, "actors", []) or [],
            preferred_directors=getattr(understanding, "directors", []) or [],
            language=getattr(understanding, "preferred_languages", ["en"])[0] if getattr(understanding, "preferred_languages", None) else None,
            keywords=getattr(understanding, "themes", []) or [],
            legacy_soft_genre=True
        )
        if getattr(understanding, "release_year_constraints", None):
            from app.services.intent_extractor import YearRange
            intent.year_range = YearRange(
                start=getattr(understanding.release_year_constraints, "start_year", None),
                end=getattr(understanding.release_year_constraints, "end_year", None)
            )
        if getattr(understanding, "excluded_genres", None):
            intent.avoid_genres = getattr(understanding, "excluded_genres", [])
            
        return await self.local_retrieval_engine.retrieve_candidates(
            original_query="",
            intent=intent,
            limit=limit
        )

    async def get_recommendations_for_movie(self, movie_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Legacy placeholder.
        """
        logger.info(f"Movie similarity recommendations requested for tmdb_id={movie_id} (limit={limit})")
        intent = RecommendationIntent(similar_movies=[str(movie_id)])
        return await self.local_retrieval_engine.retrieve_candidates(
            original_query="",
            intent=intent,
            limit=limit
        )

    async def get_recommendations_for_user(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Legacy placeholder.
        """
        logger.info(f"User recommendations requested for user_id='{user_id}' (limit={limit})")
        return []

    def _build_semantic_document(self, understanding: Any) -> str:
        """
        Legacy query builder helper.
        """
        parts = []
        if getattr(understanding, "search_intent", None):
            parts.append(f"Intent: {understanding.search_intent}")
        if getattr(understanding, "mood", None):
            parts.append(f"Mood: {understanding.mood}")
        if getattr(understanding, "genres", None):
            parts.append(f"Genres: {', '.join(understanding.genres)}")
        if getattr(understanding, "actors", None):
            parts.append(f"Starring: {', '.join(understanding.actors)}")
        if getattr(understanding, "directors", None):
            parts.append(f"Directed by: {', '.join(understanding.directors)}")
        if getattr(understanding, "reference_movies", None):
            parts.append(f"Like: {', '.join(understanding.reference_movies)}")
        if getattr(understanding, "user_preferences", None):
            parts.append(f"Preferences: {understanding.user_preferences}")
        return " | ".join(parts)

    async def _enrich_stubs_to_candidates(self, stubs: Dict[int, Dict[str, Any]], limit: int = 150) -> List[Dict[str, Any]]:
        """
        Enriches a dict of TMDb stubs with full movie details, cast, crew, etc.
        Concurrently queries the TMDbService.
        """
        if not stubs:
            return []
        
        # Sort stubs heuristic to get the most promising candidates first
        stubs_sorted = sorted(
            stubs.values(),
            key=lambda x: (x.get("popularity") or 0.0) * (x.get("vote_average") or 0.0),
            reverse=True
        )
        target_candidates = stubs_sorted[:limit]

        # Fetch detailed metadata concurrently
        tasks = [self.tmdb_service.fetch_movie_details(m.get("id")) for m in target_candidates if m.get("id")]
        details_list = await asyncio.gather(*tasks)

        candidates = []
        for details in details_list:
            if not details:
                continue
            tmdb_id = details.get("id")
            movie_dict = {
                "tmdb_id": tmdb_id,
                "title": details.get("title"),
                "original_title": details.get("original_title"),
                "overview": details.get("overview"),
                "genres": [g.get("name") for g in details.get("genres", []) if g.get("name")],
                "release_year": int(details.get("release_date", "0000")[:4]) if details.get("release_date") else None,
                "rating_value": details.get("vote_average"),
                "vote_count": details.get("vote_count"),
                "popularity": details.get("popularity"),
                "poster_path": details.get("poster_path"),
                "backdrop_path": details.get("backdrop_path"),
                "tagline": details.get("tagline"),
                "collection": details.get("belongs_to_collection", {}).get("name") if details.get("belongs_to_collection") else None,
                "keywords": [k.get("name") for k in details.get("keywords", {}).get("keywords", []) if k.get("name")],
                "cast": [m.get("name") for m in details.get("credits", {}).get("cast", [])[:10]],
                "directors": [m.get("name") for m in details.get("credits", {}).get("crew", []) if m.get("job") == "Director"],
                "production_companies": [c.get("name") for c in details.get("production_companies", []) if c.get("name")],
                "production_countries": [c.get("name") for c in details.get("production_countries", []) if c.get("name")]
            }
            candidates.append(movie_dict)
            
        return candidates


