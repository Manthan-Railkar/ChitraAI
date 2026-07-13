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
from app.vector_db.qdrant import QdrantWrapper
from app.services.embedding_service import EmbeddingService
from app.services.search_service import SearchService


class TestSearchEngine(unittest.TestCase):
    """Tests for SearchService and FastAPI search routes."""

    def setUp(self):
        # 1. Initialize real in-memory Qdrant client
        self.real_memory_client = QdrantClient(location=":memory:")
        
        # 2. Setup mock embedding service
        self.mock_embedding_service = MagicMock(spec=EmbeddingService)
        # Mock 768-dim query embedding vector
        self.mock_query_vector = np.zeros(768, dtype=np.float32)
        self.mock_query_vector[0] = 1.0  # simple indicator vector
        self.mock_embedding_service.encode_single.return_value = self.mock_query_vector
        self.mock_embedding_service.get_embedding_dimension.return_value = 768

        # 3. Create test collection in-memory
        self.wrapper = QdrantWrapper(collection_name="movies")
        self.wrapper.client = self.real_memory_client
        self.wrapper.create_collection(vector_size=768, distance_metric="Cosine")

        # 4. Populate with test movies (with different ranking signals)
        # Movie A: high semantic similarity (1.0), low popularity (10.0), low votes (100), low rating (5.0)
        # Movie B: medium similarity (0.8), high rating (9.0), high popularity (200), high votes (100000)
        # Movie C: low similarity (0.6), low rating (2.0), low popularity (0.0), low votes (0)
        
        vec_a = np.zeros(768, dtype=np.float32)
        vec_a[0] = 1.0  # cosine sim with query will be 1.0
        
        vec_b = np.zeros(768, dtype=np.float32)
        vec_b[0] = 0.8  # cosine sim with query will be 0.8
        vec_b[1] = 0.6
        
        vec_c = np.zeros(768, dtype=np.float32)
        vec_c[0] = 0.6  # cosine sim with query will be 0.6
        vec_c[1] = 0.8


        points = [
            PointStruct(
                id="00000000-0000-0000-0000-000000000001",
                vector=vec_a.tolist(),
                payload={
                    "title": "Movie A",
                    "rating_value": 5.0,
                    "popularity": 10.0,
                    "vote_count": 100,
                    "genres": ["Action"],
                    "directors": ["Dir A"]
                }
            ),
            PointStruct(
                id="00000000-0000-0000-0000-000000000002",
                vector=vec_b.tolist(),
                payload={
                    "title": "Movie B",
                    "rating_value": 9.0,
                    "popularity": 200.0,
                    "vote_count": 100000,
                    "genres": ["Sci-Fi", "Drama"],
                    "directors": ["Dir B"]
                }
            ),
            PointStruct(
                id="00000000-0000-0000-0000-000000000003",
                vector=vec_c.tolist(),
                payload={
                    "title": "Movie C",
                    "rating_value": 2.0,
                    "popularity": 0.0,
                    "vote_count": 0,
                    "genres": ["Comedy"],
                    "directors": ["Dir C"]
                }
            )
        ]
        self.real_memory_client.upsert(collection_name="movies", points=points)

        # 5. Initialize SearchService under test
        self.search_service = SearchService(self.wrapper, self.mock_embedding_service)

    def _patch_qdrant_client(self):
        """Patcher to force wrapper to use our real in-memory QdrantClient."""
        return patch("app.vector_db.qdrant.QdrantClient", return_value=self.real_memory_client)

    def _patch_services(self):
        """Patcher to inject our mock services into FastAPI endpoints."""
        return patch("app.api.routes.search.get_search_service", return_value=self.search_service)

    @patch("app.services.search_service.EmbeddingService")
    def test_search_service_reranking_math(self, mock_emb_class):
        """Verify the correctness of the combined hybrid reranked score calculation."""
        # Test SearchService directly
        import asyncio
        results = asyncio.run(self.search_service.search_movies(query="test", limit=3))
        
        self.assertEqual(len(results), 3)

        # Verify Movie B ranks first due to strong rating, popularity, and votes weights
        # Semantic B = 0.8, Rating B = 0.9, Pop B = min(1, log1p(200)/5) = min(1, 5.30/5.0) = 1.0, Votes B = min(1, log1p(100k)/15) = min(1, 11.51/15.0) = 0.767
        # Score B = 0.6 * 0.8 + 0.2 * 0.9 + 0.1 * 1.0 + 0.1 * 0.767 = 0.48 + 0.18 + 0.10 + 0.0767 = 0.8367
        
        # Semantic A = 1.0, Rating A = 0.5, Pop A = min(1, log1p(10)/5) = 0.479, Votes A = min(1, log1p(100)/15) = 0.307
        # Score A = 0.6 * 1.0 + 0.2 * 0.5 + 0.1 * 0.479 + 0.1 * 0.307 = 0.60 + 0.10 + 0.0479 + 0.0307 = 0.7786

        self.assertEqual(results[0]["title"], "Movie B")
        self.assertEqual(results[1]["title"], "Movie A")
        self.assertEqual(results[2]["title"], "Movie C")

        # Check values
        self.assertAlmostEqual(results[0]["semantic_score"], 0.8, places=4)
        self.assertTrue(results[0]["reranked_score"] > results[1]["reranked_score"])

    def test_search_service_empty_results(self):
        """Verify behavior when Qdrant returns no matches."""
        # Clear collection points
        points, _ = self.real_memory_client.scroll(collection_name="movies")
        if points:
            self.real_memory_client.delete(
                collection_name="movies",
                points_selector=[pt.id for pt in points]
            )
        
        import asyncio
        results = asyncio.run(self.search_service.search_movies(query="test", limit=10))
        self.assertEqual(results, [])


    def test_search_endpoint_success(self):
        """Validate API route returns 200, matches schema, and orders correctly."""
        client = TestClient(app)
        from app.api.deps import get_search_service
        
        app.dependency_overrides[get_search_service] = lambda: self.search_service
        
        with self._patch_qdrant_client():
            try:
                response = client.get("/api/v1/search?q=space%20adventure&limit=2")
                self.assertEqual(response.status_code, 200)
                data = response.json()
                
                self.assertEqual(data["query"], "space adventure")
                self.assertEqual(data["pagination"]["limit"], 2)
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
