"""
Unit and integration tests for the ChitraAI Final serving API Layer.
Validates autocomplete prefix checks, details endpoints, similar movies matching with soft boosting,
real-time dynamic TMDb API metadata enrichment, and standard response envelopes
(pagination and execution statistics) using a real in-memory Qdrant client database.
"""

import sys
import unittest
import asyncio
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import numpy as np
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from main import app
from app.vector_db.qdrant import QdrantWrapper
from app.services.embedding_service import EmbeddingService
from app.services.tmdb_service import TMDbService
from app.services.recommendation_service import RecommendationService
from app.services.search_service import SearchService
from app.api.deps import (
    get_qdrant_wrapper,
    get_tmdb_service,
    get_recommendation_service,
    get_search_service
)


class TestServingApiLayer(unittest.TestCase):
    """Tests for serving API routes and real-time metadata enrichment."""

    def setUp(self):
        # 1. Initialize real in-memory Qdrant client
        self.real_memory_client = QdrantClient(location=":memory:")

        # 2. Setup mock embedding service
        self.mock_embedding_service = MagicMock(spec=EmbeddingService)
        self.mock_query_vector = np.zeros(768, dtype=np.float32)
        self.mock_query_vector[0] = 1.0
        self.mock_embedding_service.encode_single.return_value = self.mock_query_vector

        # 3. Create test collection in-memory
        self.wrapper = QdrantWrapper(collection_name="movies")
        self.wrapper.client = self.real_memory_client
        self.wrapper.create_collection(vector_size=768, distance_metric="Cosine")

        # 4. Populate with test movies (with normalized vectors yielding different sims)
        self.matrix_id = "00000000-0000-0000-0000-000000000001"
        self.avatar_id = "00000000-0000-0000-0000-000000000002"
        self.tdkr_id = "00000000-0000-0000-0000-000000000003"
        self.alien_id = "00000000-0000-0000-0000-000000000004"

        vec_alien = np.zeros(768, dtype=np.float32)
        vec_alien[0] = 0.95
        vec_alien[1] = 0.31225  # mag = 1.0, sim = 0.95

        vec_matrix = np.zeros(768, dtype=np.float32)
        vec_matrix[0] = 0.90
        vec_matrix[1] = 0.43589  # mag = 1.0, sim = 0.90

        vec_tdkr = np.zeros(768, dtype=np.float32)
        vec_tdkr[0] = 0.80
        vec_tdkr[1] = 0.60  # mag = 1.0, sim = 0.80

        vec_avatar = np.zeros(768, dtype=np.float32)
        vec_avatar[0] = 0.70
        vec_avatar[1] = 0.71414  # mag = 1.0, sim = 0.70

        points = [
            PointStruct(
                id=self.matrix_id,
                vector=vec_matrix.tolist(),
                payload={
                    "title": "The Matrix",
                    "genres": ["Action", "Sci-Fi"],
                    "cast": ["Keanu Reeves", "Laurence Fishburne"],
                    "directors": ["Lana Wachowski"],
                    "release_year": 1999,
                    "rating_value": 8.7,
                    "popularity": 80.0,
                    "vote_count": 22000,
                    "tmdb_id": 603,
                    "imdb_id": "tt0133093"
                }
            ),
            PointStruct(
                id=self.avatar_id,
                vector=vec_avatar.tolist(),
                payload={
                    "title": "Avatar",
                    "genres": ["Action", "Adventure", "Sci-Fi"],
                    "cast": ["Sam Worthington", "Zoe Saldana"],
                    "directors": ["James Cameron"],
                    "release_year": 2009,
                    "rating_value": 7.8,
                    "popularity": 75.0,
                    "vote_count": 18000,
                    "tmdb_id": 19995,
                    "imdb_id": "tt0499549"
                }
            ),
            PointStruct(
                id=self.tdkr_id,
                vector=vec_tdkr.tolist(),
                payload={
                    "title": "The Dark Knight Rises",
                    "genres": ["Action", "Thriller"],
                    "cast": ["Christian Bale", "Gary Oldman"],
                    "directors": ["Christopher Nolan"],
                    "release_year": 2012,
                    "rating_value": 8.4,
                    "popularity": 90.0,
                    "vote_count": 25000,
                    "tmdb_id": 49026,
                    "imdb_id": "tt1345836"
                }
            ),
            PointStruct(
                id=self.alien_id,
                vector=vec_alien.tolist(),
                payload={
                    "title": "Alien",
                    "genres": ["Horror", "Sci-Fi"],
                    "cast": ["Sigourney Weaver", "Tom Skerritt"],
                    "directors": ["Ridley Scott"],
                    "release_year": 1979,
                    "rating_value": 8.5,
                    "popularity": 45.0,
                    "vote_count": 9000,
                    "tmdb_id": 348,
                    "imdb_id": "tt0078748"
                }
            )
        ]
        self.real_memory_client.upsert(collection_name="movies", points=points)

        # 5. Setup Mock TMDb Service & Cache Manager
        self.mock_tmdb_service = MagicMock(spec=TMDbService)
        self.mock_tmdb_service.api_key = "mock_key"
        self.mock_tmdb_service.cache = MagicMock()
        # Mock Cache to return None (meaning it triggers API simulator)
        self.mock_tmdb_service.cache.get_movie_details.return_value = None

        # Custom mock detail response
        self.mock_details = {
            "overview": "A futuristic action movie.",
            "popularity": 99.5,
            "runtime": 136,
            "poster_path": "/poster.jpg",
            "backdrop_path": "/backdrop.jpg",
            "genres": [{"name": "Action"}, {"name": "Sci-Fi"}],
            "keywords": [{"name": "cyberpunk"}],
            "videos": {
                "results": [
                    {"site": "YouTube", "type": "Trailer", "key": "abc"}
                ]
            },
            "watch/providers": {
                "results": {
                    "US": {
                        "flatrate": [{"provider_name": "Netflix"}]
                    }
                }
            },
            "release_dates": {
                "results": [
                    {
                        "iso_3166_1": "US",
                        "release_dates": [{"certification": "R"}]
                    }
                ]
            },
            "credits": {
                "cast": [{"name": "Keanu Reeves"}, {"name": "Carrie-Anne Moss"}]
            }
        }
        self.mock_tmdb_service.fetch_movie_details = AsyncMock(return_value=self.mock_details)

        # 6. Initialize recommendation service
        self.recommendation_service = RecommendationService(self.wrapper, self.mock_embedding_service)

    def test_autocomplete_endpoint(self):
        """Verify subtitle scroll matches suggestions correctly."""
        client = TestClient(app)

        with patch("app.vector_db.qdrant.QdrantClient", return_value=self.real_memory_client):
            app.dependency_overrides[get_qdrant_wrapper] = lambda: self.wrapper
            try:
                response = client.get("/api/v1/movies/autocomplete?q=Matrix&limit=5")
                self.assertEqual(response.status_code, 200)
                data = response.json()
                
                self.assertEqual(data["query"], "Matrix")
                suggestions = data["suggestions"]
                self.assertEqual(len(suggestions), 1)
                self.assertEqual(suggestions[0]["title"], "The Matrix")
                self.assertEqual(suggestions[0]["id"], self.matrix_id)
            finally:
                app.dependency_overrides.clear()

    def test_movie_details_endpoint(self):
        """Verify details fetch and merge with TMDb mock details."""
        client = TestClient(app)

        with patch("app.vector_db.qdrant.QdrantClient", return_value=self.real_memory_client):
            app.dependency_overrides[get_qdrant_wrapper] = lambda: self.wrapper
            app.dependency_overrides[get_tmdb_service] = lambda: self.mock_tmdb_service
            try:
                response = client.get(f"/api/v1/movies/{self.matrix_id}")
                self.assertEqual(response.status_code, 200)
                data = response.json()
                
                # Check envelope keys
                self.assertIn("execution_statistics", data)
                self.assertIn("movie", data)
                
                movie = data["movie"]
                self.assertEqual(movie["title"], "The Matrix")
                # Checked fields enriched from TMDb mock
                self.assertEqual(movie["poster_path"], "/poster.jpg")
                self.assertEqual(movie["trailer_url"], "https://www.youtube.com/watch?v=abc")
                self.assertEqual(movie["certification"], "R")
                self.assertEqual(movie["runtime_minutes"], 136)
                self.assertEqual(movie["streaming_providers"], ["Netflix"])
                self.assertEqual(movie["cast"], ["Keanu Reeves", "Carrie-Anne Moss"])
            finally:
                app.dependency_overrides.clear()

    def test_similar_movies_endpoint(self):
        """Verify similar movies fetching, boosting, and reranking."""
        client = TestClient(app)

        # Similar movies request for TDKR.
        # Avatar shares Action and Sci-Fi with TDKR? No, TDKR is Action, Thriller. Avatar is Action, Adventure, Sci-Fi.
        # They share "Action" genre -> boost +0.03.
        # Matrix shares "Action" with TDKR -> boost +0.03.
        # Alien shares nothing with TDKR.
        # The reasons should state the matched overlapping attributes.
        with patch("app.vector_db.qdrant.QdrantClient", return_value=self.real_memory_client):
            app.dependency_overrides[get_qdrant_wrapper] = lambda: self.wrapper
            app.dependency_overrides[get_tmdb_service] = lambda: self.mock_tmdb_service
            try:
                response = client.get(f"/api/v1/recommendations/movie/{self.tdkr_id}?limit=2")
                self.assertEqual(response.status_code, 200)
                data = response.json()
                
                self.assertEqual(data["query"], f"movie:{self.tdkr_id}")
                self.assertEqual(len(data["results"]), 2)
                
                # The source movie itself (TDKR) should be excluded
                ids = [r["id"] for r in data["results"]]
                self.assertNotIn(self.tdkr_id, ids)
                
                # Check reason structure
                first_movie = data["results"][0]
                self.assertIn("recommendation_reason", first_movie)
                self.assertIn("The Dark Knight Rises", first_movie["recommendation_reason"])
            finally:
                app.dependency_overrides.clear()

    def test_execution_statistics_and_pagination(self):
        """Verify response envelop matching pagination and statistics fields."""
        client = TestClient(app)

        mock_search_service = MagicMock(spec=SearchService)
        mock_search_service.search_movies = AsyncMock(return_value=[
            {
                "id": self.matrix_id,
                "title": "The Matrix",
                "tmdb_id": 603,
                "imdb_id": "tt0133093",
                "semantic_score": 0.9,
                "reranked_score": 0.9
            }
        ])

        with patch("app.vector_db.qdrant.QdrantClient", return_value=self.real_memory_client):
            app.dependency_overrides[get_qdrant_wrapper] = lambda: self.wrapper
            app.dependency_overrides[get_tmdb_service] = lambda: self.mock_tmdb_service
            app.dependency_overrides[get_search_service] = lambda: mock_search_service
            try:
                # Test Search Route
                response = client.get("/api/v1/search?q=space&limit=3")
                self.assertEqual(response.status_code, 200)
                data = response.json()
                
                self.assertIn("pagination", data)
                self.assertEqual(data["pagination"]["limit"], 3)
                self.assertEqual(data["pagination"]["page"], 1)
                
                self.assertIn("execution_statistics", data)
                self.assertGreater(data["execution_statistics"]["elapsed_time_ms"], 0.0)
                self.assertEqual(data["execution_statistics"]["source"], "api") # triggered API calls
            finally:
                app.dependency_overrides.clear()



if __name__ == "__main__":
    unittest.main()
