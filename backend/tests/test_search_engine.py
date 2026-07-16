"""
Unit and integration tests for the ChitraAI Semantic Search Engine.
Validates the search service reranking math, empty results, and FastAPI
endpoint behavior (successful search, schema validation, and parameter checks)
using a real in-memory Qdrant client.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from main import app
from app.services.embedding_service import EmbeddingService
from app.services.search_service import SearchService
from app.services.local_retrieval import LocalRetrievalEngine
from app.api.deps import get_search_service
import polars as pl


class TestSearchEngine(unittest.TestCase):
    """Tests for SearchService and FastAPI search routes."""

    def setUp(self):
        # 1. Setup mock embedding service
        self.mock_embedding_service = MagicMock(spec=EmbeddingService)
        # Mock 768-dim query embedding vector
        self.mock_query_vector = np.zeros(768, dtype=np.float32)
        self.mock_query_vector[0] = 1.0  # simple indicator vector
        self.mock_embedding_service.encode_single.return_value = self.mock_query_vector
        self.mock_embedding_service.get_embedding_dimension.return_value = 768

        # 2. Setup mock vectors
        vec_a = np.zeros(768, dtype=np.float32)
        vec_a[0] = 1.0  # cosine sim with query will be 1.0
        
        vec_b = np.zeros(768, dtype=np.float32)
        vec_b[0] = 0.8  # cosine sim with query will be 0.8
        vec_b[1] = 0.6
        
        vec_c = np.zeros(768, dtype=np.float32)
        vec_c[0] = 0.6  # cosine sim with query will be 0.6
        vec_c[1] = 0.8

        # 3. Create LocalRetrievalEngine with mock details
        self.local_engine = LocalRetrievalEngine(self.mock_embedding_service)
        self.local_engine.movies_df = pl.DataFrame([
            {
                "tmdb_id": 1,
                "imdb_id": "tt01",
                "movielens_id": 1,
                "wiki_page": "",
                "title": "Movie A",
                "original_title": "Movie A",
                "overview": "",
                "plot_summary": None,
                "genres": ["Action"],
                "cast": [],
                "directors": ["Dir A"],
                "writers": [],
                "runtime_minutes": 100,
                "release_year": 2000,
                "rating_value": 5.0,
                "vote_count": 100,
                "popularity": 10.0,
                "production_companies": [],
                "languages": ["en"],
                "keywords": [],
                "source_dataset": "tmdb",
                "poster_path": None,
                "backdrop_path": None,
                "trailer_url": None,
                "streaming_providers": [],
                "collection_name": None,
                "certification": None,
                "tagline": None,
                "document": ""
            },
            {
                "tmdb_id": 2,
                "imdb_id": "tt02",
                "movielens_id": 2,
                "wiki_page": "",
                "title": "Movie B",
                "original_title": "Movie B",
                "overview": "",
                "plot_summary": None,
                "genres": ["Sci-Fi", "Drama"],
                "cast": [],
                "directors": ["Dir B"],
                "writers": [],
                "runtime_minutes": 120,
                "release_year": 2005,
                "rating_value": 9.0,
                "vote_count": 100000,
                "popularity": 200.0,
                "production_companies": [],
                "languages": ["en"],
                "keywords": [],
                "source_dataset": "tmdb",
                "poster_path": None,
                "backdrop_path": None,
                "trailer_url": None,
                "streaming_providers": [],
                "collection_name": None,
                "certification": None,
                "tagline": None,
                "document": ""
            },
            {
                "tmdb_id": 3,
                "imdb_id": "tt03",
                "movielens_id": 3,
                "wiki_page": "",
                "title": "Movie C",
                "original_title": "Movie C",
                "overview": "",
                "plot_summary": None,
                "genres": ["Comedy"],
                "cast": [],
                "directors": ["Dir C"],
                "writers": [],
                "runtime_minutes": 90,
                "release_year": 2010,
                "rating_value": 2.0,
                "vote_count": 0,
                "popularity": 0.0,
                "production_companies": [],
                "languages": ["en"],
                "keywords": [],
                "source_dataset": "tmdb",
                "poster_path": None,
                "backdrop_path": None,
                "trailer_url": None,
                "streaming_providers": [],
                "collection_name": None,
                "certification": None,
                "tagline": None,
                "document": ""
            }
        ])

        # Convert to matrix
        self.local_engine.embeddings_matrix = np.array([vec_a, vec_b, vec_c], dtype=np.float32)
        norms = np.linalg.norm(self.local_engine.embeddings_matrix, axis=1, keepdims=True)
        self.local_engine.embeddings_matrix = self.local_engine.embeddings_matrix / np.where(norms == 0, 1e-12, norms)
        
        self.local_engine.tmdb_id_to_idx = {
            1: 0,
            2: 1,
            3: 2
        }

        # 4. Initialize SearchService under test
        self.search_service = SearchService(self.local_engine)

    def test_search_service_reranking_math(self):
        """Verify the correctness of the combined hybrid reranked score calculation."""
        import asyncio
        results = asyncio.run(self.search_service.search_movies(query="test", limit=3))
        
        self.assertEqual(len(results), 3)

        # Verify Movie B ranks first due to strong rating, popularity, and votes weights
        self.assertEqual(results[0]["title"], "Movie B")
        self.assertEqual(results[1]["title"], "Movie A")
        self.assertEqual(results[2]["title"], "Movie C")

        # Check values
        self.assertAlmostEqual(results[0]["semantic_score"], 0.8, places=4)
        self.assertTrue(results[0]["reranked_score"] > results[1]["reranked_score"])

    def test_search_service_empty_results(self):
        """Verify behavior when database has no matches."""
        self.local_engine.movies_df = pl.DataFrame([], schema=self.local_engine.movies_df.schema)
        import asyncio
        results = asyncio.run(self.search_service.search_movies(query="test", limit=10))
        self.assertEqual(results, [])

    def test_search_endpoint_success(self):
        """Validate API route returns 200, matches schema, and orders correctly."""
        client = TestClient(app)
        
        # Mock TMDbService to prevent real TMDb API requests from overwriting mock genres
        from app.services.tmdb_service import TMDbService
        mock_tmdb = MagicMock(spec=TMDbService)
        mock_tmdb.api_key = None
        mock_tmdb.cache = MagicMock()
        mock_tmdb.cache.get_movie_details.return_value = None
        async def mock_fetch_details(tmdb_id):
            return None
        mock_tmdb.fetch_movie_details = mock_fetch_details
        
        from app.api.deps import get_tmdb_service
        app.dependency_overrides[get_search_service] = lambda: self.search_service
        app.dependency_overrides[get_tmdb_service] = lambda: mock_tmdb
        
        try:
            response = client.get("/api/v1/search?q=space%20adventure&limit=2")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            self.assertEqual(data["query"], "space adventure")
            self.assertEqual(data["metadata"]["pagination"]["limit"], 2)
            self.assertEqual(len(data["results"]), 2)
            
            # Check ordering and metadata fields
            self.assertEqual(data["results"][0]["title"], "Movie B")
            self.assertEqual(data["results"][0]["genres"], ["Sci-Fi", "Drama"])
            self.assertEqual(data["results"][0]["directors"], ["Dir B"])
            self.assertEqual(data["results"][1]["title"], "Movie A")
        finally:
            app.dependency_overrides.clear()

    def test_search_endpoint_validation_errors(self):
        """Verify HTTP 422 Unprocessable Entity for invalid parameters."""
        client = TestClient(app)
        
        # Missing q query parameter
        response = client.get("/api/v1/search?limit=10")
        self.assertEqual(response.status_code, 422)
        
        # Query string too short (empty)
        response = client.get("/api/v1/search?q=&limit=10")
        self.assertEqual(response.status_code, 422)
        
        # Limit too large
        response = client.get("/api/v1/search?q=test&limit=200")
        self.assertEqual(response.status_code, 422)

        # Limit negative
        response = client.get("/api/v1/search?q=test&limit=-5")
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
