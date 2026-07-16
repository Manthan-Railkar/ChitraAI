import math
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import polars as pl
from loguru import logger
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.intent_extractor import RecommendationIntent


def tokenize(text: str) -> List[str]:
    """Tokenizes and normalizes text: lowercase conversion, punctuation removal, and whitespace cleanup."""
    if not text:
        return []
    # 1. Lowercase conversion
    text = text.lower()
    # 2. Punctuation removal: replace non-word/non-whitespace characters with space
    text = re.sub(r'[^\w\s]', ' ', text)
    # 3. Whitespace cleanup
    return text.split()


class StructuredFilter:
    """
    Component for applying hard filtering on movie metadata using extracted intent.
    """
    @staticmethod
    def filter_movies(df: pl.DataFrame, intent: RecommendationIntent) -> pl.DataFrame:
        lf = df.lazy()
        
        # 1. Exclusion: Avoid Genres
        if intent.avoid_genres:
            avoid_genres_lower = [g.lower() for g in intent.avoid_genres]
            lf = lf.filter(
                ~pl.col("genres").cast(pl.List(pl.String)).list.eval(pl.element().str.to_lowercase().is_in(avoid_genres_lower)).list.any()
            )
            
        # 2. Exclusion: Avoid Movies
        if intent.avoid_movies:
            avoid_movies_lower = [m.lower() for m in intent.avoid_movies]
            lf = lf.filter(
                ~pl.col("title").str.to_lowercase().is_in(avoid_movies_lower) &
                ~pl.col("original_title").str.to_lowercase().is_in(avoid_movies_lower)
            )

        # 2b. Generic Exclusions
        if hasattr(intent, "exclusions") and intent.exclusions:
            exclusions_lower = [e.lower().strip() for e in intent.exclusions if e]
            schema = lf.collect_schema()
            for exclusion in exclusions_lower:
                if "genres" in schema:
                    lf = lf.filter(
                        ~pl.col("genres").cast(pl.List(pl.String)).list.eval(pl.element().str.to_lowercase() == exclusion).list.any()
                    )
                if "cast" in schema:
                    lf = lf.filter(
                        ~pl.col("cast").cast(pl.List(pl.String)).list.eval(pl.element().str.to_lowercase() == exclusion).list.any()
                    )
                if "title" in schema:
                    lf = lf.filter(pl.col("title").str.to_lowercase() != exclusion)
                if "original_title" in schema:
                    lf = lf.filter(pl.col("original_title").str.to_lowercase() != exclusion)

        # 3. Hard Filter: Preferred Language
        if intent.language:
            lang_code = StructuredFilter.map_language_to_code(intent.language)
            lf = lf.filter(pl.col("languages").cast(pl.List(pl.String)).list.contains(lang_code))

        # 4. Hard Filter: Release Year constraints
        if intent.year_range:
            if intent.year_range.start is not None:
                lf = lf.filter(pl.col("release_year") >= intent.year_range.start)
            if intent.year_range.end is not None:
                lf = lf.filter(pl.col("release_year") <= intent.year_range.end)

        # 4b. Hard Filter: Exact Release Year constraint
        if hasattr(intent, "release_year") and intent.release_year is not None:
            lf = lf.filter(pl.col("release_year") == intent.release_year)

        # 5. Hard Filter: Preferred maximum runtime
        if intent.runtime is not None:
            lf = lf.filter(pl.col("runtime_minutes") <= intent.runtime)

        # 6. Hard Filter: Preferred Genres
        if intent.genres and not intent.legacy_soft_genre:
            genres_lower = [g.lower() for g in intent.genres]
            lf = lf.filter(
                pl.col("genres").cast(pl.List(pl.String)).list.eval(pl.element().str.to_lowercase().is_in(genres_lower)).list.any()
            )
            
        # 7. Media Type Constraints: Exclude Documentaries unless explicitly requested
        has_doc_request = any("documentary" in g.lower() for g in intent.genres) if intent.genres else False
        schema = lf.collect_schema()
        if not has_doc_request and "genres" in schema:
            lf = lf.filter(
                ~pl.col("genres").cast(pl.List(pl.String)).list.eval(pl.element().str.to_lowercase() == "documentary").list.any()
            )
            
        # 8. Media Type Constraints: Exclude TV Shows, Episodes, Box Sets, Specials, and Collections by title keywords
        exclude_patterns = r"(?i)\b(tv series|tv show|box set|season \d+|episode \d+|complete series|collection)\b"
        lf = lf.filter(
            ~pl.col("title").fill_null("").str.contains(exclude_patterns) &
            ~pl.col("original_title").fill_null("").str.contains(exclude_patterns)
        )
            
        return lf.collect()

    @staticmethod
    def map_language_to_code(lang: str) -> str:
        """Maps full language names or codes to standard ISO 639-1 code."""
        lang_lower = lang.lower().strip()
        mapping = {
            "english": "en", "french": "fr", "spanish": "es", "german": "de",
            "italian": "it", "japanese": "ja", "korean": "ko", "chinese": "zh",
            "russian": "ru", "portuguese": "pt", "hindi": "hi", "swedish": "sv"
        }
        return mapping.get(lang_lower, lang_lower[:2])


class SemanticSimilarityCalculator:
    """
    Component for calculating local cosine similarity between query and candidate embeddings.
    """
    @staticmethod
    def compute_similarities(
        query_vector: np.ndarray,
        candidate_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Computes cosine similarity between 1D query vector and 2D candidate embeddings.
        Assumes both are L2-normalized.
        """
        if candidate_embeddings.size == 0:
            return np.array([], dtype=np.float32)
            
        similarities = np.dot(candidate_embeddings, query_vector)
        return np.clip(similarities, -1.0, 1.0)


def calculate_award_score(movie: Dict[str, Any]) -> float:
    """Computes a normalized award score based on award mentions in movie metadata."""
    text_parts = []
    for field in ["overview", "plot_summary", "tagline", "title"]:
        val = movie.get(field)
        if val:
            text_parts.append(str(val))
    
    kws = movie.get("keywords")
    if kws:
        if isinstance(kws, list):
            text_parts.extend([str(k) for k in kws])
        else:
            text_parts.append(str(kws))
            
    full_text = " ".join(text_parts).lower()
    
    award_keywords = ["oscar", "academy award", "bafta", "golden globe", "cannes", "venice", "sundance"]
    matches = sum(1 for kw in award_keywords if kw in full_text)
    
    return min(1.0, matches * 0.33)


def get_improved_rating_score(movie: Dict[str, Any]) -> float:
    """Computes a Bayesian-like rating score considering both average rating and vote count."""
    rating = float(movie.get("rating_value") or 0.0)
    votes = float(movie.get("vote_count") or 0.0)
    
    C = 500.0
    baseline = 5.5
    weighted_rating = (votes * rating + C * baseline) / (votes + C)
    return weighted_rating / 10.0
    
def get_quality_rating_score(rating: float, votes: float) -> float:
    """Computes a production-grade quality rating score combining Bayesian average and log vote counts."""
    # Bayesian adjustment: push ratings with low vote counts toward the database mean (approx 6.8)
    m = 25000.0
    C_rating = 6.8
    bayesian_rating = (votes * rating + m * C_rating) / (votes + m)
    
    # Normalize vote count via log10 to prevent blockbuster saturation (max count scaled to 1.0)
    vote_factor = min(1.0, math.log10(max(1.0, votes)) / 7.0)
    
    # Blend: 85% Bayesian average rating, 15% normalized popularity weight
    return 0.85 * (bayesian_rating / 10.0) + 0.15 * vote_factor


class WeightedScorer:
    """
    Component for combining semantic similarity and metadata signals into a final score.
    """
    @staticmethod
    def score_candidates(
        movies: List[Dict[str, Any]],
        intent: RecommendationIntent,
        semantic_scores: np.ndarray
    ) -> List[Dict[str, Any]]:
        scored_movies = []
        
        pref_genres = {g.lower() for g in intent.genres}
        pref_keywords = {k.lower() for k in (intent.keywords + intent.themes + intent.moods)}
        pref_actors = {a.lower() for a in intent.preferred_actors}
        pref_directors = {d.lower() for d in intent.preferred_directors}
        
        ranking_mode = getattr(intent, "ranking_mode", "default").strip().lower()
        if ranking_mode in ["similar_movie", "similar"]:
            ranking_mode = "similar"
            
        # First compute base scores and metadata
        raw_scored = []
        for idx, movie in enumerate(movies):
            semantic_score = float(semantic_scores[idx])
            title = movie.get("title", "")
            
            # 1. Genre Overlap
            movie_genres = [g.lower() for g in (movie.get("genres") or [])]
            genre_score = 0.0
            if pref_genres and movie_genres:
                genre_score = len(set(movie_genres) & pref_genres) / len(pref_genres)
                
            # 2. Keyword/Theme Overlap
            movie_keywords = [k.lower() for k in (movie.get("keywords") or [])]
            keyword_score = 0.0
            if pref_keywords and movie_keywords:
                keyword_score = len(set(movie_keywords) & pref_keywords) / len(pref_keywords)
                
            # 3. TMDB Popularity
            pop = float(movie.get("popularity") or 0.0)
            popularity_score = min(1.0, math.log1p(pop) / 5.0)
            
            # 4. Rating Value & Votes
            rating = float(movie.get("rating_value") or 0.0)
            votes = float(movie.get("vote_count") or 0.0)
            rating_score = rating / 10.0
            
            # 5. Actor Match
            movie_cast = [a.lower() for a in (movie.get("cast") or [])]
            actor_score = 0.0
            if pref_actors and movie_cast:
                actor_score = len(set(movie_cast) & pref_actors) / len(pref_actors)
                
            # 6. Director Match
            movie_directors = [d.lower() for d in (movie.get("directors") or [])]
            director_score = 0.0
            if pref_directors and movie_directors:
                director_score = len(set(movie_directors) & pref_directors) / len(pref_directors)

            # Boosted Semantic Score
            boost = 0.0
            if pref_genres and set(movie_genres) & pref_genres:
                boost += 0.03
            if pref_actors and set(movie_cast) & pref_actors:
                boost += 0.05
            if pref_directors and set(movie_directors) & pref_directors:
                boost += 0.05
            boosted_semantic_score = semantic_score + boost

            # Extract Mood & Theme scores for MOOD ranking
            movie_keywords_lower = [k.lower() for k in (movie.get("keywords") or [])]
            movie_genres_lower = [g.lower() for g in (movie.get("genres") or [])]
            movie_overview_lower = (movie.get("overview") or "").lower()
            
            # Check Mood Match
            pref_moods = {m.lower() for m in intent.moods}
            known_moods = {"dark", "funny", "feel-good", "scary", "romantic", "intense", "thrilling", "spooky", "creepy", "hilarious", "heartwarming", "suspenseful"}
            if not pref_moods:
                pref_moods = {k.lower() for k in intent.keywords if k.lower() in known_moods}
            
            mood_score = 0.0
            if pref_moods:
                matches = sum(1 for m in pref_moods if m in movie_keywords_lower or m in movie_genres_lower or m in movie_overview_lower)
                mood_score = matches / len(pref_moods)
                
            # Check Theme Match
            pref_themes = {t.lower() for t in intent.themes}
            known_themes = {"mind-bending", "heist", "time travel", "space travel", "revenge", "dystopian", "cyberpunk", "magic"}
            if not pref_themes:
                pref_themes = {k.lower() for k in intent.keywords if k.lower() in known_themes}
                
            theme_score = 0.0
            if pref_themes:
                matches = sum(1 for t in pref_themes if t in movie_keywords_lower or t in movie_genres_lower or t in movie_overview_lower)
                theme_score = matches / len(pref_themes)

            # Select ranking score based on intent's ranking mode
            if ranking_mode == "best":
                quality_rating_score = get_quality_rating_score(rating, votes)
                vote_count_score = min(1.0, math.log1p(votes) / 12.0)
                award_score = calculate_award_score(movie)
                # Boost popularity for BEST queries
                popularity_score = min(1.0, math.log1p(pop) / 5.0)
                
                retrieval_score = (
                    0.30 * genre_score +
                    0.25 * quality_rating_score +
                    0.15 * vote_count_score +
                    0.10 * award_score +
                    0.10 * popularity_score +
                    0.10 * boosted_semantic_score
                )
            elif ranking_mode == "similar":
                retrieval_score = (
                    0.45 * boosted_semantic_score +
                    0.20 * keyword_score +
                    0.15 * director_score +
                    0.10 * actor_score +
                    0.10 * genre_score
                )
            elif ranking_mode == "mood":
                retrieval_score = (
                    0.35 * boosted_semantic_score +
                    0.25 * mood_score +
                    0.20 * theme_score +
                    0.20 * genre_score
                )
            elif ranking_mode == "discover":
                quality_rating_score = get_quality_rating_score(rating, votes)
                retrieval_score = (
                    0.40 * boosted_semantic_score +
                    0.20 * genre_score +
                    0.15 * quality_rating_score +
                    0.15 * popularity_score +
                    0.10 * keyword_score
                )
            elif ranking_mode == "classic":
                quality_rating_score = get_quality_rating_score(rating, votes)
                vote_count_score = min(1.0, math.log1p(votes) / 12.0)
                year = float(movie.get("release_year") or 1990)
                classic_year_score = max(0.0, min(1.0, (1990 - year) / 50.0))
                
                retrieval_score = (
                    0.30 * classic_year_score +
                    0.25 * quality_rating_score +
                    0.15 * vote_count_score +
                    0.15 * boosted_semantic_score +
                    0.15 * genre_score
                )
            elif ranking_mode == "underrated":
                quality_rating_score = get_quality_rating_score(rating, votes)
                vote_count_score = min(1.0, math.log1p(votes) / 12.0)
                # Reduce/penalize popularity influence for underrated queries
                underrated_pop_score = max(0.0, 1.0 - min(1.0, pop / 100.0))
                
                retrieval_score = (
                    0.30 * quality_rating_score +
                    0.25 * underrated_pop_score +
                    0.15 * vote_count_score +
                    0.15 * boosted_semantic_score +
                    0.15 * genre_score
                )
            elif ranking_mode == "recent":
                quality_rating_score = get_quality_rating_score(rating, votes)
                year = float(movie.get("release_year") or 2015)
                recent_year_score = max(0.0, min(1.0, (year - 2000) / 26.0))
                
                retrieval_score = (
                    0.30 * recent_year_score +
                    0.20 * popularity_score +
                    0.20 * boosted_semantic_score +
                    0.15 * genre_score +
                    0.15 * quality_rating_score
                )
            else:
                # Combined score formula (matching backend.md weights)
                retrieval_score = (
                    0.40 * boosted_semantic_score +
                    0.20 * genre_score +
                    0.15 * keyword_score +
                    0.15 * rating_score +
                    0.10 * popularity_score
                )

            # Step 6: Genre Constraints (mandatory)
            missing_genre_penalty = 0.0
            if pref_genres:
                # Apply a severe penalty ONLY if the movie matches NONE of the preferred genres
                if not (pref_genres & set(movie_genres)):
                    missing_genre_penalty = 0.50

            retrieval_score = max(0.0, retrieval_score - missing_genre_penalty)

            # Match counts for explanation
            matched_g = [g for g in (movie.get("genres") or []) if g.lower() in pref_genres]
            matched_a = [a for a in (movie.get("cast") or []) if a.lower() in pref_actors]
            matched_d = [d for d in (movie.get("directors") or []) if d.lower() in pref_directors]

            # Step 9: Better Recommendation Explanation
            reason = ""
            genres_phrase = " ".join(movie.get("genres")[:3])
            
            if ranking_mode == "best":
                reasons = []
                if matched_g:
                    reasons.append(f"genre ({', '.join(sorted(list(set(matched_g))))})")
                award_text = " ".join([movie.get("overview") or "", movie.get("tagline") or ""]).lower()
                has_award = any(kw in award_text for kw in ["oscar", "academy award", "bafta", "golden globe", "cannes", "venice", "sundance"])
                if has_award:
                    reasons.append("award recognition")
                if rating >= 8.0:
                    reasons.append(f"outstanding ratings ({rating}/10)")
                if reasons:
                    reason = f"Highly recommended due to its {', '.join(reasons)}."
                else:
                    reason = "One of the highest-rated films ever made with outstanding critical acclaim."
            elif ranking_mode == "similar":
                similar_movies_matched = [m for m in intent.similar_movies if m.lower() in title.lower()]
                if similar_movies_matched:
                    reason = f"Recommended because it closely matches your request for a film similar to {', '.join(similar_movies_matched)}."
                elif matched_d:
                    reason = f"Recommended because of the director ({', and '.join(sorted(list(set(matched_d))))})."
                elif matched_a:
                    reason = f"Recommended because of the shared cast ({', and '.join(sorted(list(set(matched_a))))})."
                else:
                    reason = "Recommended due to strong thematic similarity to your query."
            elif ranking_mode == "mood":
                matched_mood_themes = [mt for mt in (intent.moods + intent.themes) if mt.lower() in movie_keywords_lower or mt.lower() in movie_genres_lower]
                if matched_mood_themes:
                    reason = f"Matches the mood and theme: {', '.join(matched_mood_themes)}."
                else:
                    reason = "Matches the requested mood and tone."
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
                    reason = "This movie " + " and ".join(reason_parts) + "."
                else:
                    reason = f"Selected for its combination of {genres_phrase.lower() or 'compelling'} themes, exceptional audience ratings, and storytelling."

            # Save breakdown components to print or inspect
            components = {
                "genre_score": genre_score,
                "keyword_score": keyword_score,
                "popularity_score": popularity_score,
                "rating_score": rating_score,
                "actor_score": actor_score,
                "director_score": director_score,
                "boosted_semantic_score": boosted_semantic_score,
                "mood_score": mood_score,
                "theme_score": theme_score,
                "award_score": award_score if ranking_mode == "best" else 0.0,
                "classic_year_score": classic_year_score if ranking_mode == "classic" else 0.0,
                "underrated_pop_score": underrated_pop_score if ranking_mode == "underrated" else 0.0,
                "recent_year_score": recent_year_score if ranking_mode == "recent" else 0.0,
                "missing_genre_penalty": missing_genre_penalty
            }

            raw_scored.append({
                "movie": movie,
                "semantic_score": semantic_score,
                "boosted_semantic_score": boosted_semantic_score,
                "retrieval_score": retrieval_score,
                "reason": reason,
                "components": components
            })
            
        # Sort and apply relative confidence scores after final ranking
        raw_scored.sort(key=lambda x: x["retrieval_score"], reverse=True)
        
        for rank, item in enumerate(raw_scored):
            movie = item["movie"]
            # Relative confidence score mapping
            conf_val = max(0.75, 0.98 - rank * 0.03)
            
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
                "confidence_score": round(conf_val, 4),
                "reranked_score": round(item["retrieval_score"], 4),
                "retrieval_score": round(item["retrieval_score"], 4),
                "recommendation_reason": item["reason"],
                "score_components": item["components"]
            }
            scored_movies.append(movie_scored)
            
        return scored_movies


def apply_diversity(scored_movies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Greedily reranks scored movies to diversify the final recommendations."""
    if len(scored_movies) <= 1:
        return scored_movies
        
    diverse_list = []
    remaining = list(scored_movies)
    
    # Always keep the top choice
    diverse_list.append(remaining.pop(0))
    
    while len(diverse_list) < len(scored_movies) and remaining:
        # If we already have 3 diverse movies, we can append the rest of remaining and exit
        if len(diverse_list) >= 3:
            diverse_list.extend(remaining)
            break
            
        best_candidate_idx = 0
        best_effective_score = -9999.0
        
        # Scan a window of the top candidates to find the most diverse alternative
        scan_window = min(15, len(remaining))
        
        for i in range(scan_window):
            candidate = remaining[i]
            base_score = candidate["retrieval_score"]
            
            penalty = 0.0
            for selected in diverse_list:
                # 1. Title/Sequel Check (Franchise)
                sel_title = selected.get("title", "").lower()
                cand_title = candidate.get("title", "").lower()
                
                # Check if titles share significant words (excluding common small words)
                sel_words = set(re.findall(r'[a-z]+', sel_title)) - {"the", "a", "an", "of", "and", "in", "to", "for"}
                cand_words = set(re.findall(r'[a-z]+', cand_title)) - {"the", "a", "an", "of", "and", "in", "to", "for"}
                
                shared_words = sel_words & cand_words
                if len(shared_words) >= 2 or (len(shared_words) >= 1 and any(w in ["story", "matrix", "alien", "wars", "knight", "godfather", "trek"] for w in shared_words)):
                    penalty += 0.25
                
                # Collection Check
                sel_coll = selected.get("collection_name")
                cand_coll = candidate.get("collection_name")
                if sel_coll and cand_coll and sel_coll == cand_coll:
                    penalty += 0.30
                    
                # 2. Cast overlap check (sharing more than 2 cast members)
                sel_cast = set(c.lower() for c in (selected.get("cast") or []))
                cand_cast = set(c.lower() for c in (candidate.get("cast") or []))
                if sel_cast and cand_cast:
                    shared_cast = sel_cast & cand_cast
                    if len(shared_cast) >= 2:
                        penalty += 0.20
                        
                # 3. Director overlap check (sharing same director)
                sel_dirs = set(d.lower() for d in (selected.get("directors") or []))
                cand_dirs = set(d.lower() for d in (candidate.get("directors") or []))
                if sel_dirs and cand_dirs and (sel_dirs & cand_dirs):
                    penalty += 0.25
                    
                # 4. Release Era check (same decade / close years)
                sel_year = selected.get("release_year")
                cand_year = candidate.get("release_year")
                if sel_year and cand_year:
                    if abs(sel_year - cand_year) <= 5:
                        penalty += 0.05
                        
                # 5. Theme/Genre overlap check
                sel_genres = set(g.lower() for g in (selected.get("genres") or []))
                cand_genres = set(g.lower() for g in (candidate.get("genres") or []))
                shared_genres = sel_genres & cand_genres
                
                sel_keywords = set(k.lower() for k in (selected.get("keywords") or []))
                cand_keywords = set(k.lower() for k in (candidate.get("keywords") or []))
                shared_kws = sel_keywords & cand_keywords
                
                if len(shared_genres) >= 2 and len(shared_kws) >= 3:
                    penalty += 0.15
            
            effective_score = base_score - penalty
            if effective_score > best_effective_score:
                best_effective_score = effective_score
                best_candidate_idx = i
                
        diverse_list.append(remaining.pop(best_candidate_idx))
        
    return diverse_list


class LocalRetrievalEngine:
    """
    Main orchestration service for local dataset loading, structured filtering,
    semantic search, local BM25 keyword search, and configurable candidate score fusion.
    """
    def __init__(self, embedding_service: EmbeddingService) -> None:
        self.embedding_service = embedding_service
        self.movies_df: Optional[pl.DataFrame] = None
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.tmdb_id_to_idx: Dict[int, int] = {}
        self.tmdb_id_to_df_idx: Dict[int, int] = {}
        self.bm25: Optional[BM25Okapi] = None
        self.last_debug_report: Dict[str, Any] = {}
        
        self.processed_dir = Path(settings.PROCESSED_DATA_DIR)
        self.embeddings_dir = Path(settings.EMBEDDINGS_DIR)
        
        logger.info("LocalRetrievalEngine created.")

    def initialize(self) -> None:
        """Loads dataset and precomputed embeddings. Generates them if missing."""
        logger.info("Initializing LocalRetrievalEngine dataset & embeddings...")
        
        tmdb_parquet_path = self.processed_dir / "canonical" / "tmdb_canonical.parquet"
        if not tmdb_parquet_path.exists():
            raise FileNotFoundError(f"TMDb canonical dataset not found at {tmdb_parquet_path}. Please run rebuild_tmdb_canonical.py first.")
            
        logger.info(f"Loading TMDb canonical metadata from {tmdb_parquet_path}...")
        self.movies_df = pl.read_parquet(tmdb_parquet_path)
        logger.info(f"Loaded {self.movies_df.height:,} movies.")
        
        embeddings_path = self.embeddings_dir / "tmdb_embeddings.parquet"
        if not embeddings_path.exists():
            logger.warning(f"TMDb embeddings file not found at {embeddings_path}. Generating now locally (takes ~2 minutes)...")
            self._generate_embeddings(embeddings_path)
            
        logger.info(f"Loading TMDb precomputed embeddings from {embeddings_path}...")
        emb_df = pl.read_parquet(embeddings_path)
        
        logger.info("Converting embeddings to NumPy matrix for fast cosine similarity...")
        aligned_df = self.movies_df.select(["tmdb_id"]).join(emb_df, on="tmdb_id", how="inner")
        
        # 1. Extract raw embeddings efficiently without generating 17M+ python float objects
        raw_embs = np.vstack(aligned_df.select("embedding").to_series().to_numpy()).astype(np.float32)
        self.embeddings_matrix = raw_embs
        
        norms = np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        self.embeddings_matrix = self.embeddings_matrix / norms
        
        tmdb_ids = aligned_df.select("tmdb_id").to_series().to_numpy()
        self.tmdb_id_to_idx = {int(tid): idx for idx, tid in enumerate(tmdb_ids)}
        
        # Map tmdb_id to index in self.movies_df (to align BM25 indexes)
        self.tmdb_id_to_df_idx = {int(row["tmdb_id"]): idx for idx, row in enumerate(self.movies_df.select(["tmdb_id"]).to_dicts())}
        
        # Add columns to self.movies_df to enable O(1) index slice during retrieval
        mapping_df = pl.DataFrame({
            "tmdb_id": pl.Series(tmdb_ids, dtype=pl.Int64),
            "embedding_idx": pl.Series(range(len(tmdb_ids)), dtype=pl.Int64)
        })
        
        df_idx_series = pl.Series("df_idx", range(self.movies_df.height), dtype=pl.Int64)
        self.movies_df = self.movies_df.with_columns(df_idx_series)
        self.movies_df = self.movies_df.join(mapping_df, on="tmdb_id", how="left")
        
        logger.info(f"LocalRetrievalEngine initialized successfully. Matrix shape: {self.embeddings_matrix.shape}")
        
        # Build local BM25 index once
        self._build_bm25_index()

        # Explicitly release large temporary DataFrames and run garbage collection
        del emb_df
        del aligned_df
        import gc
        gc.collect()

    def _build_bm25_index(self) -> None:
        """Generates a local BM25 index from indexed fields of the loaded movie database."""
        if self.movies_df is None:
            logger.warning("Cannot build BM25 index: movies_df is None.")
            return

        logger.info("Building BM25 index from movie database...")
        
        # Select and build document strings entirely in Polars
        doc_df = self.movies_df.select([
            (
                pl.col("title").fill_null("") + " " +
                pl.col("overview").fill_null("") + " " +
                pl.col("tagline").fill_null("") + " " +
                pl.col("genres").cast(pl.List(pl.String)).list.join(" ").fill_null("") + " " +
                pl.col("keywords").cast(pl.List(pl.String)).list.join(" ").fill_null("") + " " +
                pl.col("directors").cast(pl.List(pl.String)).list.join(" ").fill_null("") + " " +
                pl.col("cast").cast(pl.List(pl.String)).list.slice(0, 10).list.join(" ").fill_null("") + " " +
                pl.col("plot_summary").fill_null("")
            ).alias("doc_str")
        ])
        
        doc_strings = doc_df.select("doc_str").to_series().to_list()
        tokenized_corpus = [tokenize(doc) for doc in doc_strings]
        
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index built. Documents: {self.bm25.corpus_size}, Avg doc length: {sum(self.bm25.doc_len)/len(self.bm25.doc_len):.2f}")

    def _generate_embeddings(self, output_path: Path) -> None:
        """Helper to generate embeddings for all TMDb movies on the fly and save them."""
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Building metadata documents to embed...")
        docs = []
        tmdb_ids = self.movies_df.select("tmdb_id").to_series().to_list()
        titles = self.movies_df.select("title").to_series().to_list()
        taglines = self.movies_df.select("tagline").to_series().to_list()
        overviews = self.movies_df.select("overview").to_series().to_list()
        genres_list = self.movies_df.select("genres").to_series().to_list()
        keywords_list = self.movies_df.select("keywords").to_series().to_list()
        cast_list = self.movies_df.select("cast").to_series().to_list()
        directors_list = self.movies_df.select("directors").to_series().to_list()
        
        for idx in range(len(tmdb_ids)):
            doc = build_embedding_document(
                title=titles[idx],
                tagline=taglines[idx],
                overview=overviews[idx],
                genres=genres_list[idx],
                keywords=keywords_list[idx],
                cast=cast_list[idx],
                directors=directors_list[idx]
            )
            docs.append(doc)
            
        logger.info(f"Generating embeddings for {len(docs):,} movies using {settings.EMBEDDING_MODEL}...")
        
        batch_size = 256
        embeddings_all = []
        
        start_time = time.time()
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i : i + batch_size]
            batch_embs = self.embedding_service.encode_batch(batch_docs, normalize=True)
            embeddings_all.append(batch_embs)
            
            elapsed = time.time() - start_time
            processed = min(i + batch_size, len(docs))
            speed = processed / elapsed if elapsed > 0 else 0
            eta = (len(docs) - processed) / speed if speed > 0 else 0
            logger.info(f"Embedded {processed:,}/{len(docs):,} movies | ETA: {eta/60:.1f} minutes")
            
        embeddings_matrix = np.vstack(embeddings_all)
        
        logger.info(f"Saving embeddings parquet to {output_path}...")
        emb_lists = [emb.tolist() for emb in embeddings_matrix]
        emb_df = pl.DataFrame({
            "tmdb_id": tmdb_ids,
            "embedding": emb_lists
        })
        emb_df.write_parquet(output_path)
        logger.info("Embeddings generation complete.")

    async def retrieve_candidates(self, original_query: str, intent: RecommendationIntent, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Executes local structured filtering, semantic retrieval, local BM25 keyword search,
        and configurable candidate score fusion, returning Top candidate recommendations.
        """
        if self.movies_df is None or self.embeddings_matrix is None:
            self.initialize()
            
        t_start = time.perf_counter()
        
        # Ensure embedding_idx and df_idx exist (fallback for injected mock DataFrames in tests)
        if self.movies_df is not None and "embedding_idx" not in self.movies_df.columns:
            height = self.movies_df.height
            self.tmdb_id_to_df_idx = {int(row["tmdb_id"]): idx for idx, row in enumerate(self.movies_df.select(["tmdb_id"]).to_dicts())}
            tmdb_ids_list = list(self.tmdb_id_to_idx.keys())
            mapping_df = pl.DataFrame({
                "tmdb_id": pl.Series(tmdb_ids_list, dtype=pl.Int64),
                "embedding_idx": pl.Series([self.tmdb_id_to_idx[tid] for tid in tmdb_ids_list], dtype=pl.Int64)
            })
            df_idx_series = pl.Series("df_idx", range(height), dtype=pl.Int64)
            self.movies_df = self.movies_df.with_columns(df_idx_series)
            self.movies_df = self.movies_df.join(mapping_df, on="tmdb_id", how="left")

        # Ensure BM25 index is built (fallback for injected mock DataFrames in tests)
        if self.bm25 is None:
            self._build_bm25_index()
        
        # 1. Apply Structured Hard Filters
        filtered_df = StructuredFilter.filter_movies(self.movies_df, intent)
        filtered_count = filtered_df.height
        logger.debug(f"Candidate count after hard filters: {filtered_count}")
        
        relaxed = False
        relaxation_threshold = 1 if self.movies_df.height <= 10 else min(5, limit)
        if filtered_count < relaxation_threshold and (intent.preferred_actors or intent.preferred_directors or intent.genres or intent.year_range or intent.runtime):
            logger.warning(f"Filter pool is too small ({filtered_count} candidates). Relaxing filters...")
            relaxed = True
            
            # Step 1: Relax actors and directors
            relaxed_intent = intent.model_copy()
            relaxed_intent.preferred_actors = []
            relaxed_intent.preferred_directors = []
            filtered_df = StructuredFilter.filter_movies(self.movies_df, relaxed_intent)
            filtered_count = filtered_df.height
            logger.debug(f"Count after relaxing actors/directors: {filtered_count}")
            
            # Step 2: Relax genres
            if filtered_count < relaxation_threshold and intent.genres:
                relaxed_intent.genres = []
                filtered_df = StructuredFilter.filter_movies(self.movies_df, relaxed_intent)
                filtered_count = filtered_df.height
                logger.debug(f"Count after relaxing genres: {filtered_count}")
                
            # Step 3: Relax year range and runtime
            if filtered_count < relaxation_threshold:
                relaxed_intent.year_range = None
                relaxed_intent.runtime = None
                filtered_df = StructuredFilter.filter_movies(self.movies_df, relaxed_intent)
                filtered_count = filtered_df.height
                logger.debug(f"Count after relaxing year & runtime: {filtered_count}")
                
        # Filter DataFrame to keep only movies with precomputed embeddings
        filtered_df = filtered_df.filter(pl.col("embedding_idx").is_not_null())
        
        if filtered_df.height == 0:
            logger.warning("No candidate movies match the query!")
            return []

        # 2. Get embeddings of matching candidates directly from Polars Series
        candidate_indices = filtered_df.select("embedding_idx").to_series().to_numpy()
        candidate_embs = self.embeddings_matrix[candidate_indices]
        
        # 3. Embed query intent locally for semantic retrieval
        query_doc = f"{original_query}. Genres: {', '.join(intent.genres)}. Themes: {', '.join(intent.themes)}. Moods: {', '.join(intent.moods)}. Keywords: {', '.join(intent.keywords)}."
        query_vector = self.embedding_service.encode_single(query_doc, normalize=True)
        
        # 4. Compute semantic cosine similarity locally
        semantic_scores = SemanticSimilarityCalculator.compute_similarities(query_vector, candidate_embs)

        # 5. Execute BM25 Query Processing and Search
        bm25_query_parts = []
        if original_query:
            bm25_query_parts.append(original_query)
        if intent.genres:
            bm25_query_parts.extend(intent.genres)
        if intent.moods:
            bm25_query_parts.extend(intent.moods)
        if intent.themes:
            bm25_query_parts.extend(intent.themes)
        if intent.keywords:
            bm25_query_parts.extend(intent.keywords)
        if intent.similar_movies:
            bm25_query_parts.extend(intent.similar_movies)
        if intent.preferred_actors:
            bm25_query_parts.extend(intent.preferred_actors)
        if intent.preferred_directors:
            bm25_query_parts.extend(intent.preferred_directors)

        bm25_query_str = " ".join(bm25_query_parts)
        bm25_query_tokens = tokenize(bm25_query_str)
        
        # Calculate BM25 scores for all movies
        all_bm25_scores = self.bm25.get_scores(bm25_query_tokens)
        
        # Extract BM25 scores for candidates efficiently
        candidate_df_indices = filtered_df.select("df_idx").to_series().to_numpy()
        candidate_bm25_scores = all_bm25_scores[candidate_df_indices]
        
        bm25_match_count = int(np.sum(candidate_bm25_scores > 0.0))
        max_bm25 = float(np.max(candidate_bm25_scores)) if candidate_bm25_scores.size > 0 else 0.0

        # Calculate Metadata Match score parameters
        pref_genres = [g.lower() for g in intent.genres]
        pref_keywords = [k.lower() for k in (intent.keywords + intent.themes + intent.moods)]
        pref_actors = [a.lower() for a in intent.preferred_actors]
        pref_directors = [d.lower() for d in intent.preferred_directors]
        total_signals = sum([bool(pref_genres), bool(pref_keywords), bool(pref_actors), bool(pref_directors)])

        # Define match expressions in Polars
        if pref_genres:
            genre_match_expr = pl.col("genres").cast(pl.List(pl.String)).list.eval(
                pl.element().str.to_lowercase().is_in(pref_genres)
            ).list.sum() / len(pref_genres)
        else:
            genre_match_expr = pl.lit(0.0)

        if pref_keywords:
            keyword_match_expr = pl.col("keywords").cast(pl.List(pl.String)).list.eval(
                pl.element().str.to_lowercase().is_in(pref_keywords)
            ).list.sum() / len(pref_keywords)
        else:
            keyword_match_expr = pl.lit(0.0)

        if pref_actors:
            actor_match_expr = pl.col("cast").cast(pl.List(pl.String)).list.eval(
                pl.element().str.to_lowercase().is_in(pref_actors)
            ).list.sum() / len(pref_actors)
        else:
            actor_match_expr = pl.lit(0.0)

        if pref_directors:
            director_match_expr = pl.col("directors").cast(pl.List(pl.String)).list.eval(
                pl.element().str.to_lowercase().is_in(pref_directors)
            ).list.sum() / len(pref_directors)
        else:
            director_match_expr = pl.lit(0.0)

        if total_signals > 0:
            metadata_match_expr = (genre_match_expr + keyword_match_expr + actor_match_expr + director_match_expr) / total_signals
        else:
            metadata_match_expr = pl.lit(1.0)

        # 6. Candidate Score Fusion (Semantic + BM25 + Metadata Match) with Min-Max normalization
        min_sem = float(np.min(semantic_scores)) if len(semantic_scores) > 0 else 0.0
        max_sem = float(np.max(semantic_scores)) if len(semantic_scores) > 0 else 1.0
        sem_range = max_sem - min_sem
        if len(semantic_scores) > 1 and sem_range > 0.0:
            norm_semantic_scores = (semantic_scores - min_sem) / sem_range
        elif len(semantic_scores) == 1:
            norm_semantic_scores = np.ones_like(semantic_scores)
        else:
            norm_semantic_scores = np.zeros_like(semantic_scores)

        min_bm25 = float(np.min(candidate_bm25_scores)) if len(candidate_bm25_scores) > 0 else 0.0
        max_bm25 = float(np.max(candidate_bm25_scores)) if len(candidate_bm25_scores) > 0 else 1.0
        bm25_range = max_bm25 - min_bm25
        if len(candidate_bm25_scores) > 1 and bm25_range > 0.0:
            norm_bm25_scores = (candidate_bm25_scores - min_bm25) / bm25_range
        elif len(candidate_bm25_scores) == 1:
            norm_bm25_scores = np.ones_like(candidate_bm25_scores)
        else:
            norm_bm25_scores = np.zeros_like(candidate_bm25_scores)

        norm_bm25_series = pl.Series("norm_bm25", norm_bm25_scores)
        semantic_score_series = pl.Series("semantic_score", semantic_scores)
        norm_semantic_series = pl.Series("norm_semantic", norm_semantic_scores)

        filtered_df = filtered_df.with_columns([
            semantic_score_series,
            norm_semantic_series,
            norm_bm25_series,
            metadata_match_expr.alias("metadata_match_score")
        ])
        
        filtered_df = filtered_df.with_columns(
            (
                settings.HYBRID_SEMANTIC_WEIGHT * pl.col("norm_semantic") +
                settings.HYBRID_BM25_WEIGHT * pl.col("norm_bm25") +
                settings.HYBRID_METADATA_WEIGHT * pl.col("metadata_match_score")
            ).alias("fusion_score")
        )
        
        # Capture Top 100 metadata, semantic, BM25 candidates before subsetting
        top_meta = filtered_df.sort("metadata_match_score", descending=True).head(100).to_dicts()
        top_sem = filtered_df.sort("semantic_score", descending=True).head(100).to_dicts()
        top_bm25 = filtered_df.sort("norm_bm25", descending=True).head(100).to_dicts()

        # Sort and take Top FUSION_CANDIDATES_LIMIT
        top_fused_df = filtered_df.sort("fusion_score", descending=True).head(settings.FUSION_CANDIDATES_LIMIT)
        top_fusion = top_fused_df.to_dicts()
        
        candidate_movies = top_fusion
        semantic_scores_subset = top_fused_df.select("semantic_score").to_series().to_numpy().astype(np.float32)

        # 7. Compute Weighted Scoring on the fused candidate pool
        from app.services.ranking_service import RankingService
        scored_candidates = RankingService.rank_candidates(candidate_movies, intent, semantic_scores_subset)
        
        # Apply diversity step
        diversified_candidates = RankingService.apply_diversity(scored_candidates)
        top_candidates = diversified_candidates[:limit]
        
        elapsed = round((time.perf_counter() - t_start) * 1000, 2)
        
        # Helper to format debug candidate
        def format_debug_candidate(movie: dict, final_score: float = 0.0) -> dict:
            return {
                "title": movie.get("title"),
                "tmdb_id": movie.get("tmdb_id"),
                "genres": movie.get("genres") or [],
                "semantic_score": round(float(movie.get("semantic_score") or 0.0), 4),
                "bm25_score": round(float(movie.get("norm_bm25") or 0.0), 4),
                "metadata_match_score": round(float(movie.get("metadata_match_score") or 0.0), 4),
                "rating": movie.get("rating_value") or movie.get("rating"),
                "vote_count": movie.get("vote_count"),
                "popularity": movie.get("popularity"),
                "final_score": round(float(final_score), 4)
            }

        # Build detailed debug report
        self.last_debug_report = {
            "query": original_query,
            "intent": intent.model_dump() if hasattr(intent, "model_dump") else str(intent),
            "top_100_metadata_candidates": [format_debug_candidate(m) for m in top_meta],
            "top_100_semantic_candidates": [format_debug_candidate(m) for m in top_sem],
            "top_100_bm25_candidates": [format_debug_candidate(m) for m in top_bm25],
            "candidate_fusion_output": [format_debug_candidate(m, final_score=m.get("fusion_score") or 0.0) for m in top_fusion],
            "top_20_ranked_movies": [format_debug_candidate(m, final_score=m.get("retrieval_score") or 0.0) for m in scored_candidates[:20]],
            "selected_top_3": [format_debug_candidate(m, final_score=m.get("retrieval_score") or 0.0) for m in top_candidates],
            "execution_time_ms": elapsed
        }

        # Log detailed scoring components for audit
        logger.debug("--- Scored Candidates Breakdown ---")
        for rank, movie in enumerate(top_candidates):
            comps = movie.get("score_components") or {}
            logger.debug(
                f"Rank {rank+1}: {movie.get('title')} | "
                f"Genre Score: {comps.get('genre_score', 0.0):.4f} | "
                f"Semantic Score: {comps.get('boosted_semantic_score', 0.0):.4f} | "
                f"BM25/Keyword Score: {comps.get('keyword_score', 0.0):.4f} | "
                f"Rating Score: {comps.get('rating_score', 0.0):.4f} | "
                f"Popularity Score: {comps.get('popularity_score', 0.0):.4f} | "
                f"Award Score: {comps.get('award_score', 0.0):.4f} | "
                f"Genre Penalty: {comps.get('missing_genre_penalty', 0.0):.4f} | "
                f"Final Score: {movie.get('retrieval_score', 0.0):.4f}"
            )

        # Log debug statistics
        logger.info(f"Metadata Candidates: {filtered_count}")
        logger.info(f"Semantic Candidates: {len(semantic_scores)}")
        logger.info(f"BM25 Candidates: {bm25_match_count}")
        logger.info(f"Final Candidates: {len(top_candidates)}")
        logger.info(f"Retrieval Time: {elapsed} ms")
        
        return top_candidates


def apply_diversity(scored_movies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Greedily reranks scored movies to diversify the final recommendations."""
    if len(scored_movies) <= 1:
        return scored_movies
        
    diverse_list = []
    remaining = list(scored_movies)
    
    # Always keep the top choice
    diverse_list.append(remaining.pop(0))
    
    while len(diverse_list) < len(scored_movies) and remaining:
        # If we already have 3 diverse movies, we can append the rest of remaining and exit
        if len(diverse_list) >= 3:
            diverse_list.extend(remaining)
            break
            
        best_candidate_idx = 0
        best_effective_score = -9999.0
        
        # Scan a window of the top candidates to find the most diverse alternative
        scan_window = min(15, len(remaining))
        
        for i in range(scan_window):
            candidate = remaining[i]
            base_score = candidate["retrieval_score"]
            
            penalty = 0.0
            for selected in diverse_list:
                # 1. Title/Sequel Check (Franchise)
                sel_title = selected.get("title", "").lower()
                cand_title = candidate.get("title", "").lower()
                
                # Check if titles share significant words (excluding common small words)
                sel_words = set(re.findall(r'[a-z]+', sel_title)) - {"the", "a", "an", "of", "and", "in", "to", "for"}
                cand_words = set(re.findall(r'[a-z]+', cand_title)) - {"the", "a", "an", "of", "and", "in", "to", "for"}
                
                shared_words = sel_words & cand_words
                if len(shared_words) >= 2 or (len(shared_words) >= 1 and any(w in ["story", "matrix", "alien", "wars", "knight", "godfather", "trek"] for w in shared_words)):
                    penalty += 0.25
                
                # Collection Check
                sel_coll = selected.get("collection_name")
                cand_coll = candidate.get("collection_name")
                if sel_coll and cand_coll and sel_coll == cand_coll:
                    penalty += 0.30
                    
                # 2. Cast overlap check (sharing more than 2 cast members)
                sel_cast = set(c.lower() for c in (selected.get("cast") or []))
                cand_cast = set(c.lower() for c in (candidate.get("cast") or []))
                if sel_cast and cand_cast:
                    shared_cast = sel_cast & cand_cast
                    if len(shared_cast) >= 2:
                        penalty += 0.20
                        
                # 3. Theme/Genre overlap check
                sel_genres = set(g.lower() for g in (selected.get("genres") or []))
                cand_genres = set(g.lower() for g in (candidate.get("genres") or []))
                shared_genres = sel_genres & cand_genres
                
                sel_keywords = set(k.lower() for k in (selected.get("keywords") or []))
                cand_keywords = set(k.lower() for k in (candidate.get("keywords") or []))
                shared_kws = sel_keywords & cand_keywords
                
                if len(shared_genres) >= 2 and len(shared_kws) >= 3:
                    penalty += 0.15
            
            effective_score = base_score - penalty
            if effective_score > best_effective_score:
                best_effective_score = effective_score
                best_candidate_idx = i
                
        diverse_list.append(remaining.pop(best_candidate_idx))
        
    return diverse_list


def build_embedding_document(
    title: str,
    tagline: Optional[str],
    overview: Optional[str],
    genres: List[str],
    keywords: List[str],
    cast: List[str],
    directors: List[str]
) -> str:
    """Helper function to build natural language document strings for embedding."""
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if tagline:
        parts.append(f"Tagline: {tagline}")
    if overview:
        parts.append(f"Overview: {overview}")
    if genres:
        parts.append(f"Genres: {', '.join(genres)}")
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}")
    if cast:
        parts.append(f"Cast: {', '.join(cast[:10])}")
    if directors:
        parts.append(f"Directed by: {', '.join(directors)}")
    return ". ".join(parts)
