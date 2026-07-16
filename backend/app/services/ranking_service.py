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
    def calculate_bayesian_rating(rating: float, votes: float, m: float = 50.0, C: float = 6.0) -> float:
        """
        Blends a movie's average rating with vote count to prevent low-vote items from skewing scores.
        m: minimum votes required to be considered.
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
    def rank_candidates(
        candidates: List[Dict[str, Any]],
        intent: RecommendationIntent,
        semantic_scores: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Applies profile weights, quality scores, penalties, and boosting to rank candidates.
        """
        if not candidates:
            return []

        pref_genres = {g.lower() for g in intent.genres}
        pref_keywords = {k.lower() for k in (intent.keywords + intent.themes + intent.moods)}
        pref_actors = {a.lower() for a in intent.preferred_actors}
        pref_directors = {d.lower() for d in intent.preferred_directors}
        
        ranking_mode = getattr(intent, "ranking_mode", "default").strip().lower()

        # Resolve similar movie reference names for matching
        similar_ref_titles = [t.lower() for t in intent.similar_movies]

        raw_scored = []
        for idx, movie in enumerate(candidates):
            title = movie.get("title", "")
            
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

            # 4. Rating & Vote Average (Bayesian)
            rating = float(movie.get("rating_value") or 0.0)
            votes = float(movie.get("vote_count") or 0.0)
            quality_score = RankingService.calculate_bayesian_rating(rating, votes)
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
                # Recency: scale between years 1980 and 2026
                recency_score = max(0.0, min(1.0, (release_year - 1980) / 46.0))
                # Classic: older movies (e.g. pre-1995) get higher scores
                classic_year_score = max(0.0, min(1.0, (1995 - release_year) / 50.0))

            # 7. Collection/Franchise Match
            collection_score = 0.0
            collection_name = movie.get("collection_name")
            if collection_name and any(t in collection_name.lower() for t in similar_ref_titles):
                collection_score = 1.0

            # 8. Semantic Similarity Score
            semantic_score = float(semantic_scores[idx]) if (semantic_scores is not None and idx < len(semantic_scores)) else 0.0

            # Boosted Semantic Score (used for test compatibility and default rating)
            boost_sem = 0.0
            if pref_genres and set(movie_genres) & pref_genres:
                boost_sem += 0.03
            if pref_actors and set(movie_cast) & pref_actors:
                boost_sem += 0.05
            if pref_directors and set(movie_directors) & pref_directors:
                boost_sem += 0.05
            boosted_semantic_score = semantic_score + boost_sem

            # --- DYNAMIC WEIGHTED SCORING PROFILES ---
            if ranking_mode == "best":
                # BEST: Genre (30%), Quality (25%), Vote Count (15%), Popularity (15%), Keywords (10%), Recency (5%)
                score = (
                    0.30 * genre_score +
                    0.25 * quality_score +
                    0.15 * vote_count_score +
                    0.15 * popularity_score +
                    0.10 * keyword_score +
                    0.05 * recency_score
                )
            elif ranking_mode == "similar_movie" or ranking_mode == "similar":
                # SIMILAR: Keywords (35%), Genre (25%), Collection (15%), Cast (10%), Popularity (10%), Quality (5%)
                score = (
                    0.35 * keyword_score +
                    0.25 * genre_score +
                    0.15 * collection_score +
                    0.10 * actor_score +
                    0.10 * popularity_score +
                    0.05 * quality_score
                )
            elif ranking_mode == "recent":
                # RECENT: Recency (40%), Popularity (20%), Rating (20%), Vote Count (10%), Genre (10%)
                score = (
                    0.40 * recency_score +
                    0.20 * popularity_score +
                    0.20 * quality_score +
                    0.10 * vote_count_score +
                    0.10 * genre_score
                )
            elif ranking_mode == "classic":
                # CLASSIC: Rating (30%), Vote Count (25%), Popularity (20%), Release Year Bonus (15%), Awards (10%)
                award_score = RankingService.calculate_award_score(movie)
                score = (
                    0.30 * quality_score +
                    0.25 * vote_count_score +
                    0.20 * popularity_score +
                    0.15 * classic_year_score +
                    0.10 * award_score
                )
            elif ranking_mode == "discover":
                # DISCOVER: Popularity (20%), Keywords (20%), Genre (20%), Quality (20%), Recency (20%)
                score = (
                    0.20 * popularity_score +
                    0.20 * keyword_score +
                    0.20 * genre_score +
                    0.20 * quality_score +
                    0.20 * recency_score
                )
            else: # default / general
                # Combined score formula matching original backend.md/local_retrieval.py weights
                score = (
                    0.40 * boosted_semantic_score +
                    0.20 * genre_score +
                    0.15 * keyword_score +
                    0.15 * quality_score +
                    0.10 * popularity_score
                )

            # --- CONTEXT-AWARE BOOSTING ---
            boost = 0.0
            if ranking_mode != "default":
                # Genre boost
                if pref_genres and set(movie_genres) & pref_genres:
                    boost += 0.05
                # Cast/Director boost
                if pref_actors and set(movie_cast) & pref_actors:
                    boost += 0.08
                if pref_directors and set(movie_directors) & pref_directors:
                    boost += 0.08
                # Language match boost
                if intent.language:
                    lang_match = movie.get("original_language", "").lower() == intent.language.lower()
                    if lang_match:
                        boost += 0.05
                # Similar movie title reference boost
                if any(t in title.lower() for t in similar_ref_titles):
                    boost += 0.10

            score += boost

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
            conf_percent = max(75, 98 - rank * 3) # Relative confidence: 98%, 95%, 92% ...
            
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
                "confidence_score": round(conf_percent / 100.0, 4),
                "reranked_score": round(item["score"], 4),
                "retrieval_score": round(item["score"], 4),
                "recommendation_reason": item["explanation"]
            }
            ranked_list.append(movie_scored)

        return ranked_list

    @staticmethod
    def apply_diversity(ranked_movies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Greedily diversifies recommendations to avoid franchises, duplicate actors, or same release periods in top 3."""
        if len(ranked_movies) <= 1:
            return ranked_movies

        diverse_list = []
        remaining = list(ranked_movies)

        # Always keep the top ranked movie
        diverse_list.append(remaining.pop(0))

        while len(diverse_list) < len(ranked_movies) and remaining:
            # Once we select top 3 diverse choices, we can just append the rest of the candidates and stop diversifying
            if len(diverse_list) >= 3:
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

                # Compare with already selected diverse list
                for selected in diverse_list:
                    sel_title = selected.get("title", "")
                    sel_cast = set(selected.get("cast") or [])
                    sel_directors = set(selected.get("directors") or [])
                    sel_year = selected.get("release_year")

                    # 1. Franchise overlap
                    title_tokens_sel = set(sel_title.lower().split())
                    title_tokens_cand = set(cand_title.lower().split())
                    common_tokens = title_tokens_sel & title_tokens_cand - {"the", "a", "of", "and", "in", "to", "for", "with", "part", "movie", "film"}
                    if len(common_tokens) >= 2 or (len(common_tokens) >= 1 and any(tok.isdigit() for tok in title_tokens_cand)):
                        penalty += 0.25

                    # 2. Main cast overlap (2 or more main actors in common)
                    if len(cand_cast & sel_cast) >= 2:
                        penalty += 0.20

                    # 3. Director overlap
                    if cand_directors & sel_directors:
                        penalty += 0.15

                    # 4. Release era overlap (within 3 years)
                    if cand_year and sel_year and abs(cand_year - sel_year) <= 3:
                        penalty += 0.05

                effective_score = base_score - penalty
                if effective_score > best_effective_score:
                    best_effective_score = effective_score
                    best_candidate_idx = i

            diverse_list.append(remaining.pop(best_candidate_idx))

        return diverse_list
