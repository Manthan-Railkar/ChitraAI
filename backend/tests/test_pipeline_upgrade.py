import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np
import polars as pl
from typing import Dict, Any, List

from app.services.intent_extractor import RecommendationIntent, YearRange
from app.services.local_retrieval import LocalRetrievalEngine
from app.services.recommendation_service import RecommendationService
from app.services.tmdb_service import TMDbService


class TestPipelineUpgrade(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # 1. Mock Local Retrieval Engine
        self.mock_embedding_service = MagicMock()
        # Mock encoding to return an array of shape (N, 768)
        def encode_batch_mock(docs, normalize=True):
            return np.ones((len(docs), 768), dtype=np.float32)
        self.mock_embedding_service.encode_batch.side_effect = encode_batch_mock
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
                "title": "Batman Begins",
                "original_title": "Batman Begins",
                "overview": "Bruce Wayne trains with a secret league of assassins to fight crime.",
                "plot_summary": "First part of the dark knight trilogy.",
                "genres": ["Action", "Crime", "Drama"],
                "cast": ["Christian Bale", "Liam Neeson"],
                "directors": ["Christopher Nolan"],
                "runtime_minutes": 140,
                "release_year": 2005,
                "rating_value": 8.2,
                "vote_count": 15000,
                "popularity": 120.0,
                "poster_path": "/batman_begins.jpg",
                "backdrop_path": "/batman_begins_backdrop.jpg",
                "tagline": "The legend begins",
                "keywords": ["batman", "vigilante", "hero"],
                "collection_name": "Dark Knight Collection"
            },
            {
                "tmdb_id": 103,
                "imdb_id": "tt0103",
                "title": "The Batman",
                "original_title": "The Batman",
                "overview": "In his second year of fighting crime, Batman uncovers corruption in Gotham.",
                "plot_summary": "A gritty detective take on Batman.",
                "genres": ["Action", "Crime", "Mystery", "Thriller"],
                "cast": ["Robert Pattinson", "Zoe Kravitz"],
                "directors": ["Matt Reeves"],
                "runtime_minutes": 176,
                "release_year": 2022,
                "rating_value": 7.8,
                "vote_count": 8000,
                "popularity": 180.0,
                "poster_path": "/the_batman.jpg",
                "backdrop_path": "/the_batman_backdrop.jpg",
                "tagline": "Unmask the truth",
                "keywords": ["batman", "riddler", "corruption", "detective"],
                "collection_name": "Batman Reboot Collection"
            },
            {
                "tmdb_id": 104,
                "imdb_id": "tt0104",
                "title": "Batman",
                "original_title": "Batman",
                "overview": "The Dark Knight of Gotham City begins his war on crime.",
                "plot_summary": "Tim Burton's classic Batman.",
                "genres": ["Action", "Fantasy"],
                "cast": ["Michael Keaton", "Jack Nicholson"],
                "directors": ["Tim Burton"],
                "runtime_minutes": 126,
                "release_year": 1989,
                "rating_value": 7.5,
                "vote_count": 5000,
                "popularity": 70.0,
                "poster_path": "/batman_1989.jpg",
                "backdrop_path": "/batman_1989_backdrop.jpg",
                "tagline": "Have you ever danced with the devil in the pale moonlight?",
                "keywords": ["batman", "joker", "classic"],
                "collection_name": "Burton Batman Collection"
            }
        ])

        self.local_engine.embeddings_matrix = np.ones((4, 768), dtype=np.float32)
        self.local_engine.tmdb_id_to_idx = {101: 0, 102: 1, 103: 2, 104: 3}
        self.local_engine.tmdb_id_to_df_idx = {101: 0, 102: 1, 103: 2, 104: 3}

        # 2. Mock TMDb Service
        self.mock_tmdb_service = MagicMock(spec=TMDbService)
        self.mock_tmdb_service.api_key = "mock_key"
        self.mock_tmdb_service.search_movie_by_title = AsyncMock()
        self.mock_tmdb_service.fetch_movie_details = AsyncMock()
        self.mock_tmdb_service.discover_movies = AsyncMock()
        self.mock_tmdb_service.fetch_similar = AsyncMock()
        self.mock_tmdb_service.fetch_recommendations = AsyncMock()
        self.mock_tmdb_service.resolve_person_id = AsyncMock()
        self.mock_tmdb_service.fetch_person_movie_credits = AsyncMock()

        # 3. Mock Intent Extractor
        self.mock_intent_extractor = MagicMock()
        self.mock_intent_extractor.extract_intent = AsyncMock()

        # 4. Instantiate RecommendationService
        self.recommend_service = RecommendationService(
            self.local_engine, self.mock_intent_extractor, self.mock_tmdb_service
        )

    async def test_mood_based_funnel_query(self):
        """Test a mood-based query running through the 3-stage funnel path."""
        intent = RecommendationIntent(
            intent="recommendation",
            ranking_mode="mood",
            genres=["Thriller"],
            moods=["dark", "suspenseful"],
            keywords=["keeps you on edge"]
        )
        self.mock_intent_extractor.extract_intent.return_value = intent

        # Mock TMDb Discover response with stubs
        discover_data = {
            "results": [
                {"id": 101, "title": "Inception", "popularity": 150.0, "vote_average": 8.8},
                {"id": 103, "title": "The Batman", "popularity": 180.0, "vote_average": 7.8}
            ]
        }
        self.mock_tmdb_service.discover_movies.return_value = discover_data

        # Mock TMDb fetch details responses
        async def fetch_details_mock(tmdb_id):
            for m in self.local_engine.movies_df.to_dicts():
                if m["tmdb_id"] == tmdb_id:
                    # Match structure returned by TMDb API details endpoint
                    return {
                        "id": tmdb_id,
                        "title": m["title"],
                        "original_title": m["original_title"],
                        "overview": m["overview"],
                        "genres": [{"id": 1, "name": g} for g in m["genres"]],
                        "release_date": f"{m['release_year']}-01-01",
                        "vote_average": m["rating_value"],
                        "vote_count": m["vote_count"],
                        "popularity": m["popularity"],
                        "poster_path": m["poster_path"],
                        "backdrop_path": m["backdrop_path"],
                        "tagline": m["tagline"],
                        "belongs_to_collection": {"id": 999, "name": m["collection_name"]} if m["collection_name"] else None,
                        "keywords": {"keywords": [{"id": i, "name": k} for i, k in enumerate(m["keywords"])]},
                        "credits": {
                            "cast": [{"name": actor} for actor in m["cast"]],
                            "crew": [{"job": "Director", "name": dir_name} for dir_name in m["directors"]]
                        },
                        "production_companies": [],
                        "production_countries": []
                    }
            return None

        self.mock_tmdb_service.fetch_movie_details.side_effect = fetch_details_mock

        # Run recommendation
        with patch("app.services.recommendation_service.enrich_movie_with_tmdb", side_effect=lambda m, s: m):
            ret_intent, results = await self.recommend_service.recommend_movies_from_query(
                "dark suspenseful thriller that keeps you on edge", limit=10
            )

        self.assertEqual(ret_intent.ranking_mode, "mood")
        self.assertGreater(len(results), 0)
        
        # Verify 3-stage report has been populated
        report = self.local_engine.last_debug_report
        self.assertEqual(report["pipeline_pathway"], "funnel")
        self.assertEqual(report["retrieval_path"], "TMDb")
        self.assertIn("stage_1_broad_retrieval", report)
        self.assertIn("stage_2_semantic_narrowing", report)
        self.assertIn("stage_3_final_ranking", report)
        self.assertIn("funnel_summary", report)

    async def test_director_specific_strict_query(self):
        """Test director query routing through strict person filter branch."""
        intent = RecommendationIntent(
            intent="recommendation",
            ranking_mode="default",
            preferred_directors=["Christopher Nolan"],
            strict_person_filter=True
        )
        self.mock_intent_extractor.extract_intent.return_value = intent
        self.mock_tmdb_service.resolve_person_id.return_value = 500

        # Mock Nolan filmography response
        self.mock_tmdb_service.fetch_person_movie_credits.return_value = {
            "crew": [
                {"id": 101, "title": "Inception", "job": "Director", "popularity": 150.0},
                {"id": 102, "title": "Batman Begins", "job": "Director", "popularity": 120.0}
            ]
        }

        # Reuse fetch_details mock
        async def fetch_details_mock(tmdb_id):
            for m in self.local_engine.movies_df.to_dicts():
                if m["tmdb_id"] == tmdb_id:
                    return {
                        "id": tmdb_id,
                        "title": m["title"],
                        "original_title": m["original_title"],
                        "overview": m["overview"],
                        "genres": [{"id": 1, "name": g} for g in m["genres"]],
                        "release_date": f"{m['release_year']}-01-01",
                        "vote_average": m["rating_value"],
                        "vote_count": m["vote_count"],
                        "popularity": m["popularity"],
                        "poster_path": m["poster_path"],
                        "backdrop_path": m["backdrop_path"],
                        "tagline": m["tagline"],
                        "belongs_to_collection": None,
                        "keywords": {"keywords": [{"id": i, "name": k} for i, k in enumerate(m["keywords"])]},
                        "credits": {
                            "cast": [{"name": actor} for actor in m["cast"]],
                            "crew": [{"job": "Director", "name": dir_name} for dir_name in m["directors"]]
                        },
                        "production_companies": [],
                        "production_countries": []
                    }
            return None
        self.mock_tmdb_service.fetch_movie_details.side_effect = fetch_details_mock

        with patch("app.services.recommendation_service.enrich_movie_with_tmdb", side_effect=lambda m, s: m):
            ret_intent, results = await self.recommend_service.recommend_movies_from_query(
                "Christopher Nolan movies", limit=10
            )

        self.assertEqual(len(results), 2)
        for m in results:
            self.assertIn("Christopher Nolan", m["directors"])

        # Check debug report
        report = self.local_engine.last_debug_report
        self.assertEqual(report["pipeline_pathway"], "strict_person")

    async def test_exact_single_movie_lookup_hardening(self):
        """Test exact movie lookup fallback hardening for ambiguous titles."""
        intent = RecommendationIntent(
            intent="movie_lookup",
            similar_movies=["The Batman"]
        )
        self.mock_intent_extractor.extract_intent.return_value = intent

        # Force TMDb search to fail to test local database fallback hardening
        self.mock_tmdb_service.search_movie_by_title.return_value = None

        with patch("app.services.recommendation_service.enrich_movie_with_tmdb", side_effect=lambda m, s: m):
            ret_intent, results = await self.recommend_service.recommend_movies_from_query(
                "The Batman", limit=1
            )

        self.assertEqual(len(results), 1)
        # Even though "Batman" (1989) contains "Batman" and "Batman Begins" contains "Batman",
        # "The Batman" (2022) must be returned because it has 100% exact string similarity match or the highest ratio.
        self.assertEqual(results[0]["title"], "The Batman")

    async def test_similar_to_x_query_routing(self):
        """Test similar query routing through similarity engine branch."""
        intent = RecommendationIntent(
            intent="recommendation",
            ranking_mode="similar",
            similar_movies=["Inception"]
        )
        self.mock_intent_extractor.extract_intent.return_value = intent

        self.mock_tmdb_service.search_movie_by_title.return_value = 101

        # Mock TMDb fetch similar + recommendations responses
        self.mock_tmdb_service.fetch_similar.return_value = {
            "results": [
                {"id": 102, "title": "Batman Begins", "popularity": 120.0, "vote_average": 8.2}
            ]
        }
        self.mock_tmdb_service.fetch_recommendations.return_value = {
            "results": [
                {"id": 103, "title": "The Batman", "popularity": 180.0, "vote_average": 7.8}
            ]
        }

        # Reuse fetch_details mock
        async def fetch_details_mock(tmdb_id):
            for m in self.local_engine.movies_df.to_dicts():
                if m["tmdb_id"] == tmdb_id:
                    return {
                        "id": tmdb_id,
                        "title": m["title"],
                        "original_title": m["original_title"],
                        "overview": m["overview"],
                        "genres": [{"id": 1, "name": g} for g in m["genres"]],
                        "release_date": f"{m['release_year']}-01-01",
                        "vote_average": m["rating_value"],
                        "vote_count": m["vote_count"],
                        "popularity": m["popularity"],
                        "poster_path": m["poster_path"],
                        "backdrop_path": m["backdrop_path"],
                        "tagline": m["tagline"],
                        "belongs_to_collection": None,
                        "keywords": {"keywords": [{"id": i, "name": k} for i, k in enumerate(m["keywords"])]},
                        "credits": {
                            "cast": [{"name": actor} for actor in m["cast"]],
                            "crew": [{"job": "Director", "name": dir_name} for dir_name in m["directors"]]
                        },
                        "production_companies": [],
                        "production_countries": []
                    }
            return None
        self.mock_tmdb_service.fetch_movie_details.side_effect = fetch_details_mock

        with patch("app.services.recommendation_service.enrich_movie_with_tmdb", side_effect=lambda m, s: m):
            ret_intent, results = await self.recommend_service.recommend_movies_from_query(
                "movies similar to Inception", limit=10
            )

        self.assertGreater(len(results), 0)
        report = self.local_engine.last_debug_report
        self.assertEqual(report["pipeline_pathway"], "similarity")
