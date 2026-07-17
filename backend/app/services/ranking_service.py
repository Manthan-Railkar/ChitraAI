import math
from typing import Dict, Any, List, Optional
from loguru import logger
from app.services.intent_extractor import RecommendationIntent

class RankingService:
    """
    Dedicated production-grade ranking engine for ChitraAI.
    Calculates multi-dimensional ranking signals, applies dynamic profile weights,
    quality-based Bayesian scoring, context-aware boosting, and diversity filtering.
    """

    @staticmethod
    def calculate_bayesian_rating(rating: float, votes: float, m: float = 1000.0, C: float = 6.5) -> float:
        """
        Blends a movie's average rating with vote count to prevent low-vote items from skewing scores.
        m: minimum votes required to be considered. (Increased to 1000.0 for stronger production rating correction)
        C: database mean rating fallback.
        """
        if votes + m == 0:
            return C / 10.0
        return ((votes * rating) + (m * C)) / (votes + m) / 10.0


    @staticmethod
    def calculate_award_score(movie: Dict[str, Any]) -> float:
        """Scans title, tagline, overview, and keywords for major awards."""
        award_text = " ".join([
            movie.get("title") or "",
            movie.get("tagline") or "",
            movie.get("overview") or "",
            " ".join(movie.get("keywords") or [])
        ]).lower()
        
        has_award = any(kw in award_text for kw in ["oscar", "academy award", "bafta", "cannes", "golden globe", "venice", "sundance"])
        return 1.0 if has_award else 0.0

    @staticmethod
    def passes_hard_constraints(movie: Dict[str, Any], intent: RecommendationIntent) -> bool:
        """
        Applies strict filters to candidate movies BEFORE they enter scoring/ranking.
        If a movie violates a mandatory constraint (genre, language, director, actor,
        runtime, era, country, avoid lists), returns False.
        """
        def normalize_genre(g: str) -> str:
            g_low = g.lower().strip()
            if g_low in ("sci-fi", "science fiction"):
                return "science fiction"
            if g_low in ("tv movie", "tv show"):
                return "tv movie"
            return g_low

        title = movie.get("title") or movie.get("original_title") or "Unknown"

        # --- BUG #7: Candidate Quality Validation ---
        # 1. Status Check
        status = movie.get("status")
        if isinstance(status, str):
            if status.lower() in ("planned", "rumored", "post production", "in production", "cancelled"):
                logger.debug(f"[Audit Log] Rejected candidate '{title}' - unreleased or planned status: '{status}'")
                return False
                
        # 2. Release Date Check
        release_date = movie.get("release_date")
        release_year = movie.get("release_year")
        if not release_date and not release_year:
            logger.debug(f"[Audit Log] Rejected candidate '{title}' - missing release date and year.")
            return False
            
        # 3. Overview Check
        overview = movie.get("overview")
        if not overview or not isinstance(overview, str) or len(overview.strip()) < 10:
            logger.debug(f"[Audit Log] Rejected candidate '{title}' - missing or extremely short overview.")
            return False
            
        # 4. Empty Genres Check
        genres = movie.get("genres")
        if not genres:
            logger.debug(f"[Audit Log] Rejected candidate '{title}' - empty genres.")
            return False
            
        # 5. Vote Count Check
        vote_count = movie.get("vote_count")
        if vote_count is not None:
            try:
                if float(vote_count) < 5:
                    logger.debug(f"[Audit Log] Rejected candidate '{title}' - vote count ({vote_count}) below threshold.")
                    return False
            except (ValueError, TypeError):
                logger.debug(f"[Audit Log] Rejected candidate '{title}' - invalid vote count ({vote_count}).")
                return False

        # --- Standard Constraints (with audit logging) ---
        # 1. Excluded Genres Check
        movie_genres = [g.lower() for g in (movie.get("genres") or [])]
        movie_genres_norm = {normalize_genre(g) for g in movie_genres}
        avoid_genres = {normalize_genre(g) for g in intent.avoid_genres}
        if avoid_genres and (avoid_genres & movie_genres_norm):
            logger.debug(f"[Audit Log] Rejected candidate '{title}' - contains avoided genres: {avoid_genres & movie_genres_norm}")
            return False

        # 2. Excluded Movies Check
        title_lower = title.lower()
        if intent.avoid_movies:
            avoid_titles = {t.lower() for t in intent.avoid_movies if t}
            if any(at in title_lower for at in avoid_titles):
                logger.debug(f"[Audit Log] Rejected candidate '{title}' - matches avoid list movie title.")
                return False

        # 3. Explicit Genre Constraint
        explicit_genres = getattr(intent, "explicit_genres", None)
        if explicit_genres is None:
            explicit_genres = intent.genres
        pref_genres = {normalize_genre(g) for g in explicit_genres}
        if pref_genres and movie_genres_norm:
            if not (pref_genres & movie_genres_norm):
                logger.debug(f"[Audit Log] Rejected candidate '{title}' - does not match explicit genre constraint. Pref: {pref_genres}, Movie: {movie_genres_norm}")
                return False

        # 4. Language Constraint
        if intent.language:
            lang_lower = intent.language.lower().strip()
            lang_map = {
                "korean": "ko", "japanese": "ja", "english": "en", "french": "fr",
                "german": "de", "spanish": "es", "italian": "it", "chinese": "zh",
                "cantonese": "cn", "hindi": "hi", "russian": "ru", "swedish": "sv"
            }
            target_code = None
            if lang_lower in lang_map:
                target_code = lang_map[lang_lower]
            elif len(lang_lower) == 2 and lang_lower in set(lang_map.values()):
                target_code = lang_lower
                
            if target_code:
                movie_langs = movie.get("languages") or []
                if not isinstance(movie_langs, list):
                    movie_langs = [movie_langs]
                movie_langs = [str(l).lower() for l in movie_langs]
                
                orig_lang = movie.get("original_language")
                if orig_lang:
                    movie_langs.append(orig_lang.lower())
                    
                if movie_langs and target_code not in movie_langs:
                    logger.debug(f"[Audit Log] Rejected candidate '{title}' - language mismatch. Target: {target_code}, Movie: {movie_langs}")
                    return False

        # 5. Release Era Constraint
        # Check release_year
        if release_year:
            if intent.year_range:
                if intent.year_range.start and release_year < intent.year_range.start:
                    logger.debug(f"[Audit Log] Rejected candidate '{title}' - release year {release_year} < start {intent.year_range.start}")
                    return False
                if intent.year_range.end and release_year > intent.year_range.end:
                    logger.debug(f"[Audit Log] Rejected candidate '{title}' - release year {release_year} > end {intent.year_range.end}")
                    return False
            if intent.release_year:
                if abs(release_year - intent.release_year) > 3:
                    logger.debug(f"[Audit Log] Rejected candidate '{title}' - release year {release_year} outside ±3 tolerance of target {intent.release_year}")
                    return False

        # 6. Explicit Director Constraint
        if intent.preferred_directors:
            movie_dirs = [d.lower() for d in (movie.get("directors") or [])]
            pref_dirs = {d.lower() for d in intent.preferred_directors}
            if movie_dirs and not (pref_dirs & set(movie_dirs)):
                logger.debug(f"[Audit Log] Rejected candidate '{title}' - director mismatch. Pref: {pref_dirs}, Movie: {movie_dirs}")
                return False

        # 7. Explicit Actor Constraint
        if intent.preferred_actors:
            movie_cast = [a.lower() for a in (movie.get("cast") or [])]
            pref_actors = {a.lower() for a in intent.preferred_actors}
            if movie_cast and not (pref_actors & set(movie_cast)):
                logger.debug(f"[Audit Log] Rejected candidate '{title}' - actor mismatch. Pref: {pref_actors}, Movie: {movie_cast}")
                return False

        # 8. Country Constraint
        if intent.country:
            movie_countries = [c.lower() for c in (movie.get("production_countries") or [])]
            if movie_countries and not any(intent.country.lower() in mc for mc in movie_countries):
                logger.debug(f"[Audit Log] Rejected candidate '{title}' - country mismatch. Target: {intent.country}, Movie: {movie_countries}")
                return False

        # 9. Runtime Constraint
        if intent.runtime and movie.get("runtime_minutes"):
            if movie["runtime_minutes"] > (intent.runtime + 10):
                logger.debug(f"[Audit Log] Rejected candidate '{title}' - runtime {movie['runtime_minutes']} exceeds maximum limit {intent.runtime + 10}")
                return False

        # 10. Family Safety Constraint
        if intent.family_safety == "family_safe":
            if any(g in ["Horror", "Thriller", "Crime"] for g in movie.get("genres", [])):
                if "Horror" in movie.get("genres", []):
                    logger.debug(f"[Audit Log] Rejected candidate '{title}' - family safe check: genre is Horror.")
                    return False
            if movie.get("adult"):
                logger.debug(f"[Audit Log] Rejected candidate '{title}' - family safe check: adult movie.")
                return False
            movie_kws = {k.lower() for k in (movie.get("keywords") or [])}
            mature_indicators = {"r-rated", "nudity", "gore", "slasher", "mature", "nsfw", "erotic"}
            if mature_indicators & movie_kws:
                logger.debug(f"[Audit Log] Rejected candidate '{title}' - family safe check: contains mature keywords: {mature_indicators & movie_kws}")
                return False

        # 11. TV Series / Special Exclusions
        exclude_patterns = ["tv series", "tv show", "box set", "complete series", "season ", "episode "]
        if any(pat in title_lower for pat in exclude_patterns):
            logger.debug(f"[Audit Log] Rejected candidate '{title}' - TV series exclusion pattern matched.")
            return False

        # 12. Documentary Exclusions (unless requested)
        has_doc_request = any("documentary" in g.lower() for g in intent.genres)
        if not has_doc_request and "Documentary" in movie.get("genres", []):
            logger.debug(f"[Audit Log] Rejected candidate '{title}' - Documentary exclusion.")
            return False

        return True


    @staticmethod
    def calculate_match_percentage(
        movie: Dict[str, Any],
        intent: RecommendationIntent,
        semantic_score: float
    ) -> float:
        """
        Calculates a true, objective match percentage based on multiple relevance vectors:
        semantic similarity, genre overlap, keyword/theme overlap, actor/director matches,
        and studio/country criteria. Mapped to a user-friendly range [0.65, 0.99].
        """
        def normalize_genre(g: str) -> str:
            g_low = g.lower().strip()
            if g_low in ("sci-fi", "science fiction"):
                return "science fiction"
            if g_low in ("tv movie", "tv show"):
                return "tv movie"
            return g_low

        pref_genres = {normalize_genre(g) for g in intent.genres}
        pref_keywords = {k.lower() for k in (intent.keywords + intent.themes + intent.moods)}
        pref_actors = {a.lower() for a in intent.preferred_actors}
        pref_directors = {d.lower() for d in intent.preferred_directors}

        # 1. Semantic Similarity Match
        norm_sem = min(1.0, max(0.0, (semantic_score - 0.15) / 0.50))

        # 2. Genre Overlap Match
        movie_genres = [g.lower() for g in (movie.get("genres") or [])]
        movie_genres_norm = {normalize_genre(g) for g in movie_genres}
        genre_ratio = len(movie_genres_norm & pref_genres) / len(pref_genres) if pref_genres else 1.0

        # 3. Keyword / Theme Match
        movie_keywords = [k.lower() for k in (movie.get("keywords") or [])]
        kw_ratio = len(set(movie_keywords) & pref_keywords) / len(pref_keywords) if pref_keywords else 1.0

        # 4. Director / Actor Match
        dir_match = 1.0 if not pref_directors or (set([d.lower() for d in (movie.get("directors") or [])]) & pref_directors) else 0.0
        act_match = 1.0 if not pref_actors or (set([a.lower() for a in (movie.get("cast") or [])]) & pref_actors) else 0.0

        # 5. Studio Match (if requested)
        studio_match = 1.0
        if intent.studio:
            prod_companies = [c.lower() for c in movie.get("production_companies", [])]
            studio_match = 1.0 if any(intent.studio.lower() in c for c in prod_companies) else 0.0

        # 6. Country Match (if requested)
        country_match = 1.0
        if intent.country:
            movie_countries = [c.lower() for c in movie.get("production_countries", [])]
            country_match = 1.0 if any(intent.country.lower() in c for c in movie_countries) else 0.0

        # 7. Era Match (if requested)
        era_match = 1.0
        if intent.release_year and movie.get("release_year"):
            era_diff = abs(movie["release_year"] - intent.release_year)
            era_match = max(0.0, 1.0 - (era_diff / 10.0))

        # Compute raw weighted score
        relevance_score = (
            0.35 * norm_sem +
            0.20 * genre_ratio +
            0.15 * kw_ratio +
            0.10 * dir_match +
            0.10 * act_match +
            0.05 * studio_match +
            0.05 * country_match
        )
        if intent.release_year:
            relevance_score = (relevance_score * 0.9) + (era_match * 0.1)

        # Map relevance [0.0, 1.0] -> user-friendly match percentage [0.65, 0.99]
        match_percentage = 0.65 + (relevance_score * 0.34)
        return round(min(0.99, max(0.65, match_percentage)), 4)

    @staticmethod
    def rank_candidates(
        candidates: List[Dict[str, Any]],
        intent: RecommendationIntent,
        semantic_scores: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Applies profile weights, quality scores, penalties, and boosting to rank candidates.
        """
        # --- BUG #9: Input Validation & Graceful Recovery ---
        if not candidates:
            logger.warning("[Ranking Input Validation] Candidate movies list is empty. Returning empty list.")
            return []

        try:
            # Normalize genres if they are not canonical
            intent.genres = RecommendationIntent._normalize_genre_list(intent.genres or [])
            intent.avoid_genres = RecommendationIntent._normalize_genre_list(intent.avoid_genres or [])
            
            if not isinstance(intent.preferred_actors, list):
                intent.preferred_actors = [intent.preferred_actors] if intent.preferred_actors else []
            if not isinstance(intent.preferred_directors, list):
                intent.preferred_directors = [intent.preferred_directors] if intent.preferred_directors else []
            if not isinstance(intent.similar_movies, list):
                intent.similar_movies = [intent.similar_movies] if intent.similar_movies else []
            if not isinstance(intent.keywords, list):
                intent.keywords = [intent.keywords] if intent.keywords else []
            if not isinstance(intent.themes, list):
                intent.themes = [intent.themes] if intent.themes else []
            if not isinstance(intent.moods, list):
                intent.moods = [intent.moods] if intent.moods else []
        except Exception as e:
            logger.error(f"[Ranking Input Validation] Error recovering input metadata: {e}. Attempting fallback schema recovery...")
            intent.genres = intent.genres or []
            intent.avoid_genres = intent.avoid_genres or []
            intent.preferred_actors = intent.preferred_actors or []
            intent.preferred_directors = intent.preferred_directors or []
            intent.similar_movies = intent.similar_movies or []
            intent.keywords = intent.keywords or []
            intent.themes = intent.themes or []
            intent.moods = intent.moods or []

        pref_genres = {g.lower() for g in intent.genres}
        pref_keywords = {k.lower() for k in (intent.keywords + intent.themes + intent.moods)}
        pref_actors = {a.lower() for a in intent.preferred_actors}
        pref_directors = {d.lower() for d in intent.preferred_directors}
        
        ranking_mode = getattr(intent, "ranking_mode", "default").strip().lower()
        similar_ref_titles = [t.lower() for t in intent.similar_movies]

        # --- DYNAMIC RETRIEVAL & WEIGHT GENERATION ---
        w_semantic = 0.20
        w_quality = 0.20
        w_vote_count = 0.10
        w_popularity = 0.10
        w_genre = 0.15
        w_keyword = 0.10
        w_actor = 0.05
        w_director = 0.05
        w_classic = 0.0
        w_recency = 0.0
        w_award = 0.0
        w_collection = 0.05

        # Adjust based on popularity preferences
        if intent.popularity_preference == "hidden_gem" or intent.popularity_preference == "niche":
            w_popularity = -0.30
            w_quality = 0.40
            w_vote_count = -0.10
        elif intent.popularity_preference == "popular":
            w_popularity = 0.35
            w_vote_count = 0.20
            w_quality = 0.15

        # Adjust based on critical acclaim preferences
        if intent.critical_acclaim_preference == "high":
            w_quality = 0.40
            w_vote_count = 0.20

        # Adjust based on novelty/recency preferences
        if intent.novelty_preference == "classic":
            w_classic = 0.30
            w_quality = 0.25
        elif intent.novelty_preference in ("recent", "new_release", "upcoming"):
            w_recency = 0.35
            w_popularity = 0.15

        # Adjust based on awards preferences
        if intent.awards_preference and intent.awards_preference != "neutral":
            w_award = 0.30
            w_quality = 0.25

        # Adjust based on preferred actors and directors
        if intent.preferred_actors:
            w_actor = 0.30
        if intent.preferred_directors:
            w_director = 0.30

        # Adjust based on keywords/themes specific queries
        if intent.themes or intent.moods or intent.complexity == "mind_bending":
            w_keyword = 0.25
            w_semantic = 0.25

        # Compile positive weights and normalize them
        pos_weights = {
            "semantic": w_semantic,
            "quality": w_quality,
            "vote_count": w_vote_count if w_vote_count > 0 else 0.0,
            "popularity": w_popularity if w_popularity > 0 else 0.0,
            "genre": w_genre,
            "keyword": w_keyword,
            "actor": w_actor,
            "director": w_director,
            "classic": w_classic,
            "recency": w_recency,
            "award": w_award,
            "collection": w_collection
        }
        total_pos = sum(pos_weights.values())
        if total_pos > 0:
            for k in pos_weights:
                pos_weights[k] /= total_pos

        # Extract negative penalties from negative weights
        neg_pop_penalty = abs(w_popularity) if w_popularity < 0 else 0.0
        neg_votes_penalty = abs(w_vote_count) if w_vote_count < 0 else 0.0

        raw_scored = []
        for idx, movie in enumerate(candidates):
            title = movie.get("title") or movie.get("original_title") or ""
            
            # --- CALCULATE BASE SIGNAL SCORES (0.0 to 1.0) ---

            # 1. Genre Match Score
            movie_genres = [g.lower() for g in (movie.get("genres") or [])]
            genre_score = len(set(movie_genres) & pref_genres) / len(pref_genres) if pref_genres else 1.0

            # 2. Keyword/Theme Match Score
            movie_keywords = [k.lower() for k in (movie.get("keywords") or [])]
            keyword_score = len(set(movie_keywords) & pref_keywords) / len(pref_keywords) if pref_keywords else 1.0

            # 3. Popularity Score (Log normalized)
            pop = float(movie.get("popularity") or 0.0)
            popularity_score = min(1.0, math.log1p(pop) / 6.0)

            # 4. Rating & Vote Average (Bayesian with dynamic pull)
            rating = float(movie.get("rating_value") or 0.0)
            votes = float(movie.get("vote_count") or 0.0)
            
            # Use smaller m for hidden gems to avoid drowning them out
            m_pull = 150.0 if intent.popularity_preference in ("hidden_gem", "niche") else 1000.0
            quality_score = RankingService.calculate_bayesian_rating(rating, votes, m=m_pull)
            vote_count_score = min(1.0, math.log1p(votes) / 12.0)

            # 5. Actor & Director Matches
            movie_cast = [a.lower() for a in (movie.get("cast") or [])]
            actor_score = len(set(movie_cast) & pref_actors) / len(pref_actors) if pref_actors else 0.0

            movie_directors = [d.lower() for d in (movie.get("directors") or [])]
            director_score = len(set(movie_directors) & pref_directors) / len(pref_directors) if pref_directors else 0.0

            # 6. Recency / Release Year Bonus
            release_year = movie.get("release_year")
            recency_score = 0.0
            classic_year_score = 0.0
            if release_year:
                recency_score = max(0.0, min(1.0, (release_year - 1980) / 46.0))
                classic_year_score = max(0.0, min(1.0, (1995 - release_year) / 50.0))

            # 7. Collection/Franchise Match
            collection_score = 0.0
            collection_name = movie.get("collection_name") or movie.get("collection")
            if collection_name and any(t in collection_name.lower() for t in similar_ref_titles):
                collection_score = 1.0

            # 8. Semantic Similarity Score
            semantic_score = semantic_scores[idx] if (semantic_scores is not None and idx < len(semantic_scores)) else 0.0

            # Boosted Semantic Score
            boost_sem = 0.0
            if pref_genres and set(movie_genres) & pref_genres:
                boost_sem += 0.03
            if pref_actors and set(movie_cast) & pref_actors:
                boost_sem += 0.05
            if pref_directors and set(movie_directors) & pref_directors:
                boost_sem += 0.05
            boosted_semantic_score = semantic_score + boost_sem

            # 9. Awards
            award_score = RankingService.calculate_award_score(movie)

            # 10. Studio/Company Match
            studio_score = 0.0
            if intent.studio and movie.get("production_companies"):
                prod_companies = [c.lower() for c in movie.get("production_companies", [])]
                if any(intent.studio.lower() in c for c in prod_companies):
                    studio_score = 1.0

            # 11. Country Match
            country_score = 0.0
            if intent.country and movie.get("production_countries"):
                movie_countries = [c.lower() for c in movie.get("production_countries", [])]
                if any(intent.country.lower() in c for c in movie_countries):
                    country_score = 1.0

            # 12. Pacing & Complexity Match
            complexity_score = 0.0
            if intent.complexity == "mind_bending":
                complexity_score = len(set(movie_keywords) & {"mind-bending", "time travel", "illusion", "reality", "twist", "psychological", "puzzle"}) / 3.0
                complexity_score = min(1.0, complexity_score)
            
            pacing_score = 0.0
            if intent.pacing == "slow_burn":
                pacing_score = 1.0 if any(k in movie_keywords for k in ["slow burn", "slow-burn", "atmospheric", "character study", "meditative"]) else 0.0
            elif intent.pacing == "fast_paced":
                pacing_score = 1.0 if any(k in movie_keywords for k in ["fast paced", "fast-paced", "action packed", "action-packed", "quick paced", "thrill ride", "adrenaline"]) else 0.0

            # --- COMPILE FINAL SCORE ---
            score = (
                pos_weights["semantic"] * boosted_semantic_score +
                pos_weights["quality"] * quality_score +
                pos_weights["vote_count"] * vote_count_score +
                pos_weights["popularity"] * popularity_score +
                pos_weights["genre"] * genre_score +
                pos_weights["keyword"] * keyword_score +
                pos_weights["actor"] * actor_score +
                pos_weights["director"] * director_score +
                pos_weights["classic"] * classic_year_score +
                pos_weights["recency"] * recency_score +
                pos_weights["award"] * award_score +
                pos_weights["collection"] * collection_score
            )

            # --- GRADED CONTEXT BOOSTS (capped to 0.25 max) ---
            boost_components = {}
            boost_components["studio"] = 0.15 * studio_score
            boost_components["country"] = 0.10 * country_score
            boost_components["pacing"] = 0.08 * pacing_score
            boost_components["complexity"] = 0.10 * complexity_score

            # Genre match boost (graded by count)
            genre_boost = 0.0
            if pref_genres and movie_genres:
                matching_genres = set(movie_genres) & pref_genres
                genre_boost = min(0.10, 0.04 * len(matching_genres))
            boost_components["genre"] = genre_boost

            # Keyword/theme match boost (graded by count)
            kw_boost = 0.0
            if pref_keywords and movie_keywords:
                matching_kws = set(movie_keywords) & pref_keywords
                kw_boost = min(0.10, 0.025 * len(matching_kws))
            boost_components["keyword"] = kw_boost

            # Cast/Director boost (graded by match ratio)
            actor_boost = 0.0
            if pref_actors and movie_cast:
                actor_match_ratio = len(set(movie_cast) & pref_actors) / len(pref_actors)
                actor_boost = 0.08 * actor_match_ratio
            boost_components["actor"] = actor_boost

            dir_boost = 0.0
            if pref_directors and movie_directors:
                dir_match_ratio = len(set(movie_directors) & pref_directors) / len(pref_directors)
                dir_boost = 0.08 * dir_match_ratio
            boost_components["director"] = dir_boost

            # Language match boost
            lang_boost = 0.0
            if intent.language:
                lang_match = movie.get("original_language", "").lower() == intent.language.lower()
                if lang_match:
                    lang_boost = 0.05
            boost_components["language"] = lang_boost

            # Runtime proximity boost (graded)
            rt_boost = 0.0
            if intent.runtime and movie.get("runtime_minutes"):
                runtime_diff = abs(movie["runtime_minutes"] - intent.runtime)
                if runtime_diff <= 15:
                    rt_boost = 0.05 * (1.0 - runtime_diff / 15.0)
            boost_components["runtime"] = rt_boost

            # Release period boost (graded)
            period_boost = 0.0
            if intent.release_year and release_year:
                year_diff = abs(release_year - intent.release_year)
                if year_diff <= 5:
                    period_boost = 0.05 * (1.0 - year_diff / 5.0)
            boost_components["release_period"] = period_boost

            # Referenced movie title & collection boost
            ref_movie_boost = 0.0
            if any(t in title.lower() for t in similar_ref_titles):
                ref_movie_boost = 0.15
            boost_components["ref_movie"] = ref_movie_boost

            coll_boost = 0.0
            ref_collections = [c.lower() for c in getattr(intent, "ref_collections", []) if c]
            cand_collection = movie.get("collection_name") or movie.get("collection")
            if cand_collection and ref_collections:
                if cand_collection.lower() in ref_collections:
                    coll_boost = 0.10
            boost_components["ref_collection"] = coll_boost

            total_raw_boost = sum(boost_components.values())
            boost = total_raw_boost
            if total_raw_boost > 0.25:
                scale_factor = 0.25 / total_raw_boost
                for k in boost_components:
                    boost_components[k] *= scale_factor
                boost = 0.25

            score += boost

            # Apply negative penalties
            if neg_pop_penalty > 0:
                score -= neg_pop_penalty * popularity_score
            if neg_votes_penalty > 0:
                score -= neg_votes_penalty * vote_count_score

            # --- PENALTIES ---
            penalty = 0.0
            # Wrong genre penalty (matches none of the preferred genres)
            if pref_genres and not (pref_genres & set(movie_genres)):
                penalty += 0.50
            # Adult content penalty (unless requested)
            if movie.get("adult") and "adult" not in pref_genres:
                penalty += 0.80
            # Poor rating penalty
            if rating > 0.0 and rating < 4.0:
                penalty += 0.30
            # Very low vote count penalty
            if votes > 0 and votes < 10:
                penalty += 0.20

            score = max(0.0, score - penalty)

            # --- RECOMMENDATION EXPLANATION GENERATION ---
            # Match counts for explanation compatibility
            matched_g = [g for g in (movie.get("genres") or []) if g.lower() in pref_genres]
            matched_a = [a for a in (movie.get("cast") or []) if a.lower() in pref_actors]
            matched_d = [d for d in (movie.get("directors") or []) if d.lower() in pref_directors]

            genres_phrase = ", ".join(movie.get("genres")[:2]) if movie.get("genres") else "compelling"

            if ranking_mode == "best":
                explanation = f"Highly rated {genres_phrase} movie with exceptional audience rating of {rating}/10."
            elif ranking_mode == "similar":
                matched_refs = [ref for ref in intent.similar_movies if ref.lower() in title.lower()]
                if matched_refs:
                    explanation = f"Recommended because it closely matches your request for a film similar to {matched_refs[0]}."
                else:
                    explanation = f"Matches the thematic style, genres ({genres_phrase}), and tone of your query."
            elif ranking_mode == "classic":
                explanation = f"A critically acclaimed classic {genres_phrase} film released in {release_year} with {int(votes):,} votes."
            else:
                # Default explanation structure for tests compatibility
                reason_parts = []
                if matched_g:
                    genres_str = ", ".join(sorted(list(set(matched_g))))
                    reason_parts.append(f"matches preferred genre(s) ({genres_str})")
                if matched_a:
                    actors_str = ", ".join(sorted(list(set(matched_a))))
                    reason_parts.append(f"features actor(s) you like ({actors_str})")
                if matched_d:
                    directors_str = ", ".join(sorted(list(set(matched_d))))
                    reason_parts.append(f"features director(s) you like ({directors_str})")
                    
                if reason_parts:
                    explanation = "This movie " + " and ".join(reason_parts) + "."
                else:
                    explanation = f"Matches your interest in {genres_phrase} themes with strong audience appeal."

            raw_scored.append({
                "movie": movie,
                "score": score,
                "semantic_score": semantic_score,
                "boosted_semantic_score": boosted_semantic_score,
                "explanation": explanation
            })

        # Sort by score descending
        raw_scored.sort(key=lambda x: x["score"], reverse=True)

        # Build final formatted recommendation list with relative confidence scores
        ranked_list = []
        for rank, item in enumerate(raw_scored):
            movie = item["movie"]
            match_percentage = RankingService.calculate_match_percentage(movie, intent, item["semantic_score"])
            
            movie_scored = {
                "id": str(movie.get("tmdb_id")),
                "tmdb_id": movie.get("tmdb_id"),
                "title": movie.get("title"),
                "original_title": movie.get("original_title"),
                "overview": movie.get("overview"),
                "genres": movie.get("genres") or [],
                "release_year": movie.get("release_year"),
                "rating_value": movie.get("rating_value"),
                "popularity": movie.get("popularity"),
                "keywords": movie.get("keywords") or [],
                "cast": movie.get("cast") or [],
                "directors": movie.get("directors") or [],
                "poster_path": movie.get("poster_path"),
                "backdrop_path": movie.get("backdrop_path"),
                "tagline": movie.get("tagline"),
                "semantic_score": round(item["semantic_score"], 4),
                "boosted_semantic_score": round(item["boosted_semantic_score"], 4),
                "confidence_score": match_percentage,
                "reranked_score": round(item["score"], 4),
                "retrieval_score": round(item["score"], 4),
                "recommendation_reason": item["explanation"]
            }
            ranked_list.append(movie_scored)

        if ranked_list:
            top_cand = ranked_list[0]
            logger.info(
                f"[Audit Log] Ranking Completed. Top candidate: '{top_cand['title']}' "
                f"(confidence: {top_cand['confidence_score']:.4f}, "
                f"reranked score: {top_cand['reranked_score']:.4f}, "
                f"reason: {top_cand['recommendation_reason']})"
            )

        return ranked_list

    @staticmethod
    def apply_diversity(ranked_movies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Greedily diversifies recommendations to avoid franchises, duplicate actors, or same release periods in top N."""
        if len(ranked_movies) <= 1:
            return ranked_movies

        diverse_list = []
        remaining = list(ranked_movies)

        # Always keep the top ranked movie
        diverse_list.append(remaining.pop(0))

        # Diversify the full top-N (up to 10 slots) instead of only top 3
        diversify_slots = min(10, len(ranked_movies))

        while len(diverse_list) < len(ranked_movies) and remaining:
            # Once we fill diversify_slots diverse choices, append the rest and stop
            if len(diverse_list) >= diversify_slots:
                diverse_list.extend(remaining)
                break

            best_candidate_idx = 0
            best_effective_score = -9999.0

            # Scan a window of the top candidates
            scan_window = min(15, len(remaining))

            for i in range(scan_window):
                candidate = remaining[i]
                base_score = candidate["retrieval_score"]
                penalty = 0.0

                cand_title = candidate.get("title", "")
                cand_cast = set(candidate.get("cast") or [])
                cand_directors = set(candidate.get("directors") or [])
                cand_year = candidate.get("release_year")
                cand_collection = candidate.get("collection") or candidate.get("collection_name")

                # Compare with already selected diverse list
                for selected in diverse_list:
                    sel_title = selected.get("title", "")
                    sel_cast = set(selected.get("cast") or [])
                    sel_directors = set(selected.get("directors") or [])
                    sel_year = selected.get("release_year")
                    sel_collection = selected.get("collection") or selected.get("collection_name")

                    # 1. Collection/Franchise overlap
                    if cand_collection and sel_collection and cand_collection.lower() == sel_collection.lower():
                        penalty += 0.35

                    # 2. Franchise title tokens overlap
                    title_tokens_sel = set(sel_title.lower().split())
                    title_tokens_cand = set(cand_title.lower().split())
                    common_tokens = title_tokens_sel & title_tokens_cand - {"the", "a", "of", "and", "in", "to", "for", "with", "part", "movie", "film"}
                    if len(common_tokens) >= 2 or (len(common_tokens) >= 1 and any(tok.isdigit() for tok in title_tokens_cand)):
                        penalty += 0.25

                    # 3. Main cast overlap (2 or more main actors in common)
                    if len(cand_cast & sel_cast) >= 2:
                        penalty += 0.20

                    # 4. Director overlap
                    if cand_directors & sel_directors:
                        penalty += 0.15

                    # 5. Release era overlap (within 3 years)
                    if cand_year and sel_year and abs(cand_year - sel_year) <= 3:
                        penalty += 0.05

                effective_score = base_score - penalty
                if effective_score > best_effective_score:
                    best_effective_score = effective_score
                    best_candidate_idx = i

            diverse_list.append(remaining.pop(best_candidate_idx))

        return diverse_list
