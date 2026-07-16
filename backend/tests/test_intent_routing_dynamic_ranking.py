import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np
import polars as pl
from typing import Dict, Any, List

from app.services.intent_extractor import RecommendationIntent, YearRange
from app.services.local_retrieval import (
    LocalRetrievalEngine,
    WeightedScorer,
    calculate_award_score,
    get_improved_rating_score,
    apply_diversity
)
from app.services.recommendation_service import RecommendationService
from app.services.tmdb_service import TMDbService


class TestIntentRoutingAndDynamicRanking(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # 1. Mock Local Retrieval Engine
        self.mock_embedding_service = MagicMock()
        self.mock_embedding_service.encode_single.return_value = np.ones(768, dtype=np.float32)
        
        self.local_engine = LocalRetrievalEngine(self.mock_embedding_service)
        
        # Build mock dataset containing test movies
        self.local_engine.movies_df = pl.DataFrame([
            {
                "tmdb_id": 101,
                "imdb_id": "tt0101",
                "title": "Inception",
                "original_title": "Inception",
                "overview": "A thief who steals corporate secrets through the use of dream-sharing technology.",
                "plot_summary": "Academy Award winning film about dreams.",
                "genres": ["Action", "Sci-Fi", "Thriller"],
                "cast": ["Leonardo DiCaprio", "Joseph Gordon-Levitt"],
                "directors": ["Christopher Nolan"],
                "runtime_minutes": 148,
                "release_year": 2010,
                "rating_value": 8.8,
                "vote_count": 20000,
                "popularity": 150.0,
                "poster_path": "/inception.jpg",
                "backdrop_path": "/inception_backdrop.jpg",
                "tagline": "Your mind is the scene of the crime",
                "keywords": ["dreams", "heist", "mind-bending"],
                "collection_name": "Inception Collection"
            },
            {
                "tmdb_id": 102,
                "imdb_id": "tt0102",
                "title": "Inception Sequel",
                "original_title": "Inception Sequel",
                "overview": "Another dream heist adventure.",
                "plot_summary": "Sequel to the famous Oscar winning heist film.",
                "genres": ["Action", "Sci-Fi", "Thriller"],
                "cast": ["Leonardo DiCaprio", "Joseph Gordon-Levitt"],
                "directors": ["Christopher Nolan"],
                "runtime_minutes": 150,
                "release_year": 2013,
                "rating_value": 8.0,
                "vote_count": 10000,
                "popularity": 120.0,
                "poster_path": "/inception2.jpg",
                "backdrop_path": "/inception2_backdrop.jpg",
                "tagline": "Dream bigger",
                "keywords": ["dreams", "heist"],
                "collection_name": "Inception Collection"
            },
            {
                "tmdb_id": 103,
                "imdb_id": "tt0103",
                "title": "Toy Story",
                "original_title": "Toy Story",
                "overview": "A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him.",
                "plot_summary": "Award-winning Pixar animation masterpiece.",
                "genres": ["Animation", "Adventure", "Comedy"],
                "cast": ["Tom Hanks", "Tim Allen"],
                "directors": ["John Lasseter"],
                "runtime_minutes": 81,
                "release_year": 1995,
                "rating_value": 8.3,
                "vote_count": 15000,
                "popularity": 80.0,
                "poster_path": "/toystory.jpg",
                "backdrop_path": "/toystory_backdrop.jpg",
                "tagline": "The toys are back in town",
                "keywords": ["toys", "friendship", "buddy comedy"],
                "collection_name": "Toy Story Collection"
            }
        ])
        
        self.local_engine.embeddings_matrix = np.ones((3, 768), dtype=np.float32)
        self.local_engine.tmdb_id_to_idx = {101: 0, 102: 1, 103: 2}
        self.local_engine.tmdb_id_to_df_idx = {101: 0, 102: 1, 103: 2}
        
        # 2. Mock TMDb Service
        self.mock_tmdb_service = MagicMock(spec=TMDbService)
        self.mock_tmdb_service.api_key = "mock_key"
        
        # 3. Mock Intent Extractor
        self.mock_intent_extractor = MagicMock()
        
        # 4. Instantiate RecommendationService
        self.recommend_service = RecommendationService(
            self.local_engine, self.mock_intent_extractor, self.mock_tmdb_service
        )

    def test_award_score_calculation(self):
        """Verify that award score is correctly computed based on keywords in metadata fields."""
        # Movie 101 has 'Academy Award' in plot_summary
        movie_101 = self.local_engine.movies_df.to_dicts()[0]
        score_101 = calculate_award_score(movie_101)
        self.assertGreater(score_101, 0.0)
        self.assertAlmostEqual(score_101, 0.33, places=2)

        # Movie 103 has 'Award-winning' in plot_summary but doesn't match standard award names (Oscars, BAFTA etc.)
        movie_103 = self.local_engine.movies_df.to_dicts()[2]
        score_103 = calculate_award_score(movie_103)
        self.assertEqual(score_103, 0.0)

    def test_improved_rating_score(self):
        """Verify the Bayesian average rating formula weights score by vote count."""
        movie_low_votes = {"rating_value": 9.5, "vote_count": 5}
        movie_high_votes = {"rating_value": 8.5, "vote_count": 10000}
        
        score_low = get_improved_rating_score(movie_low_votes)
        score_high = get_improved_rating_score(movie_high_votes)
        
        # Movie with only 5 votes is regressed toward the baseline of 5.5, so its score is significantly lower than 9.5/10
        self.assertLess(score_low, 0.95)
        # Movie with 10,000 votes maintains its score close to 8.5/10 (0.85)
        self.assertGreater(score_high, 0.80)

    def test_dynamic_ranking_profiles(self):
        """Verify that different ranking modes use separate scoring weighting profiles."""
        movies = self.local_engine.movies_df.to_dicts()
        semantic_scores = np.array([0.9, 0.85, 0.7], dtype=np.float32)
        
        # Mode: BEST
        intent_best = RecommendationIntent(ranking_mode="best")
        scored_best = WeightedScorer.score_candidates(movies, intent_best, semantic_scores)
        
        # Mode: SIMILAR_MOVIE
        intent_similar = RecommendationIntent(ranking_mode="similar_movie")
        scored_similar = WeightedScorer.score_candidates(movies, intent_similar, semantic_scores)
        
        # The scores/ordering can differ between best and similar movie profiles
        self.assertEqual(len(scored_best), 3)
        self.assertEqual(len(scored_similar), 3)

    def test_diversity_step(self):
        """Verify that diversity reranking prunes or demotes sequels/franchises in the top 3."""
        # Setup scored movies where index 0 and 1 are Inception & Inception Sequel (same collection/franchise)
        scored_movies = [
            {
                "title": "Inception",
                "collection_name": "Inception Collection",
                "genres": ["Action", "Sci-Fi"],
                "cast": ["Leonardo DiCaprio"],
                "retrieval_score": 0.95
            },
            {
                "title": "Inception Sequel",
                "collection_name": "Inception Collection",
                "genres": ["Action", "Sci-Fi"],
                "cast": ["Leonardo DiCaprio"],
                "retrieval_score": 0.94
            },
            {
                "title": "Toy Story",
                "collection_name": "Toy Story Collection",
                "genres": ["Animation"],
                "cast": ["Tom Hanks"],
                "retrieval_score": 0.85
            }
        ]
        
        diversified = apply_diversity(scored_movies)
        
        # The sequel (Inception Sequel) should be demoted, making Toy Story (different franchise) rise to index 1
        self.assertEqual(diversified[0]["title"], "Inception")
        self.assertEqual(diversified[1]["title"], "Toy Story")
        self.assertEqual(diversified[2]["title"], "Inception Sequel")

    async def test_movie_lookup_bypass(self):
        """Verify that movie_lookup intent bypasses recommendation and hits TMDb directly."""
        self.mock_intent_extractor.extract_intent = AsyncMock(return_value=RecommendationIntent(
            intent="movie_lookup",
            similar_movies=["Inception"]
        ))
        
        # Mock TMDb API response
        mock_details = {
            "title": "Inception",
            "overview": "A dream thief details.",
            "release_date": "2010-07-16",
            "runtime": 148,
            "vote_average": 8.8,
            "vote_count": 20000,
            "popularity": 150.0,
            "genres": [{"name": "Action"}, {"name": "Sci-Fi"}],
            "credits": {
                "crew": [{"job": "Director", "name": "Christopher Nolan"}],
                "cast": [{"name": "Leonardo DiCaprio"}]
            }
        }
        self.mock_tmdb_service.search_movie_by_title.return_value = 101
        self.mock_tmdb_service.fetch_movie_details.return_value = mock_details
        
        # Mock cache/enrichment behavior to return the movie_dict
        with patch("app.services.recommendation_service.enrich_movie_with_tmdb", side_effect=lambda m, s: m):
            intent, results = await self.recommend_service.recommend_movies_from_query("tell me about Inception")
            
            self.assertEqual(intent.intent, "movie_lookup")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["title"], "Inception")
            self.assertEqual(results[0]["directors"], ["Christopher Nolan"])
            self.assertEqual(results[0]["cast"], ["Leonardo DiCaprio"])
            self.assertEqual(results[0]["release_year"], 2010)

    async def test_movie_lookup_local_fallback(self):
        """Verify that if TMDb lookup fails, we fall back to local database search."""
        self.mock_intent_extractor.extract_intent = AsyncMock(return_value=RecommendationIntent(
            intent="movie_lookup",
            similar_movies=["Toy Story"]
        ))
        
        # Mock TMDb to return no movie found
        self.mock_tmdb_service.search_movie_by_title.return_value = None
        self.mock_tmdb_service.fetch_movie_details.return_value = None
        
        # Mock enrichment to do nothing
        with patch("app.services.recommendation_service.enrich_movie_with_tmdb", side_effect=lambda m, s: m):
            intent, results = await self.recommend_service.recommend_movies_from_query("Toy Story lookup")
            
            self.assertEqual(intent.intent, "movie_lookup")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["title"], "Toy Story")
            self.assertEqual(results[0]["tmdb_id"], 103)


if __name__ == "__main__":
    unittest.main()
