"""
Unit and integration tests for the ChitraAI Hybrid Recommendation Engine.
Validates structured query builder, hard filters (exclusions & years), soft boosting,
dynamic reason generation, and FastAPI recommendations route endpoint behavior
using a real in-memory Qdrant client.
"""

import sys
import unittest
import asyncio
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
from app.services.recommendation_service import RecommendationService
from app.services.gemini_service import QueryUnderstandingResult, YearConstraints
from app.api.deps import get_recommendation_service, get_gemini_service


class TestHybridRecommendation(unittest.TestCase):
    """Tests for RecommendationService and FastAPI recommendations route."""

    def setUp(self):
        # 1. Initialize real in-memory Qdrant client
        self.real_memory_client = QdrantClient(location=":memory:")

        # 2. Setup mock embedding service
        self.mock_embedding_service = MagicMock(spec=EmbeddingService)
        # Mock 768-dim query embedding vector
        self.mock_query_vector = np.zeros(768, dtype=np.float32)
        self.mock_query_vector[0] = 1.0  # simple indicator vector
        self.mock_embedding_service.encode_single.return_value = self.mock_query_vector

        # 3. Create test collection in-memory
        self.wrapper = QdrantWrapper(collection_name="movies")
        self.wrapper.client = self.real_memory_client
        self.wrapper.create_collection(vector_size=768, distance_metric="Cosine")

        # 4. Populate with test movies (with different vectors and attributes)
        # We construct normalized vectors to yield exact base cosine similarities:
        # Alien: sim = 0.9
        # The Matrix: sim = 0.8
        # Toy Story: sim = 0.7
        # The Dark Knight Rises: sim = 0.6
        
        vec_alien = np.zeros(768, dtype=np.float32)
        vec_alien[0] = 0.9
        vec_alien[1] = 0.43589  # magnitude = sqrt(0.81 + 0.19) = 1.0
        
        vec_matrix = np.zeros(768, dtype=np.float32)
        vec_matrix[0] = 0.8
        vec_matrix[1] = 0.6  # magnitude = 1.0
        
        vec_toy_story = np.zeros(768, dtype=np.float32)
        vec_toy_story[0] = 0.7
        vec_toy_story[1] = 0.71414  # magnitude = 1.0
        
        vec_tdkr = np.zeros(768, dtype=np.float32)
        vec_tdkr[0] = 0.6
        vec_tdkr[1] = 0.8  # magnitude = 1.0

        points = [
            PointStruct(
                id="00000000-0000-0000-0000-000000000001",
                vector=vec_matrix.tolist(),
                payload={
                    "title": "The Matrix",
                    "genres": ["Action", "Sci-Fi"],
                    "cast": ["Keanu Reeves", "Laurence Fishburne"],
                    "directors": ["Lana Wachowski", "Lilly Wachowski"],
                    "release_year": 1999,
                    "rating_value": 8.7,
                    "popularity": 80.0,
                    "vote_count": 22000
                }
            ),
            PointStruct(
                id="00000000-0000-0000-0000-000000000002",
                vector=vec_toy_story.tolist(),
                payload={
                    "title": "Toy Story",
                    "genres": ["Animation", "Comedy"],
                    "cast": ["Tom Hanks", "Tim Allen"],
                    "directors": ["John Lasseter"],
                    "release_year": 1995,
                    "rating_value": 8.3,
                    "popularity": 60.0,
                    "vote_count": 15000
                }
            ),
            PointStruct(
                id="00000000-0000-0000-0000-000000000003",
                vector=vec_tdkr.tolist(),
                payload={
                    "title": "The Dark Knight Rises",
                    "genres": ["Action", "Thriller"],
                    "cast": ["Christian Bale", "Gary Oldman"],
                    "directors": ["Christopher Nolan"],
                    "release_year": 2012,
                    "rating_value": 8.4,
                    "popularity": 90.0,
                    "vote_count": 25000
                }
            ),
            PointStruct(
                id="00000000-0000-0000-0000-000000000004",
                vector=vec_alien.tolist(),
                payload={
                    "title": "Alien",
                    "genres": ["Horror", "Sci-Fi"],
                    "cast": ["Sigourney Weaver", "Tom Skerritt"],
                    "directors": ["Ridley Scott"],
                    "release_year": 1979,
                    "rating_value": 8.5,
                    "popularity": 45.0,
                    "vote_count": 9000
                }
            )
        ]
        self.real_memory_client.upsert(collection_name="movies", points=points)

        # 5. Initialize RecommendationService under test
        self.recommend_service = RecommendationService(self.wrapper, self.mock_embedding_service)

    def test_query_builder_document(self):
        """Verify structured JSON maps correctly to a text search document."""
        understanding = QueryUnderstandingResult(
            search_intent="search",
            mood="spooky",
            genres=["Horror", "Sci-Fi"],
            actors=["Sigourney Weaver"],
            directors=["Ridley Scott"],
            reference_movies=["Alien"],
            user_preferences="PG-13 only"
        )
        doc = self.recommend_service._build_semantic_document(understanding)
        
        self.assertIn("Intent: search", doc)
        self.assertIn("Mood: spooky", doc)
        self.assertIn("Genres: Horror, Sci-Fi", doc)
        self.assertIn("Starring: Sigourney Weaver", doc)
        self.assertIn("Directed by: Ridley Scott", doc)
        self.assertIn("Like: Alien", doc)
        self.assertIn("Preferences: PG-13 only", doc)

    def test_hard_filters_exclusion_and_years(self):
        """Verify hard filters prune excluded genres and year violations."""
        # Query: release after 1996, but exclude Horror
        understanding = QueryUnderstandingResult(
            search_intent="search",
            excluded_genres=["Horror"],
            release_year_constraints=YearConstraints(start_year=1996)
        )
        
        results = asyncio.run(self.recommend_service.recommend_movies_from_understanding(understanding, limit=10))
        
        # Toy Story (1995) is pruned (< 1996)
        # Alien (1979) is pruned (< 1996 and Horror)
        # Only The Matrix (1999) and Dark Knight Rises (2012) should remain
        titles = [r["title"] for r in results]
        self.assertEqual(len(results), 2)
        self.assertIn("The Matrix", titles)
        self.assertIn("The Dark Knight Rises", titles)
        self.assertNotIn("Toy Story", titles)
        self.assertNotIn("Alien", titles)

    def test_soft_boosting(self):
        """Verify matching genres/actors/directors boosts semantic score."""
        # Query: Action genre, starring Keanu Reeves, directed by John Lasseter
        # Movie A (The Matrix): Matches Action (+0.03), Keanu Reeves (+0.05). Base sim = 0.8. Boosted sim = 0.88.
        # Movie B (Toy Story): Matches John Lasseter (+0.05). Base sim = 0.7. Boosted sim = 0.75.
        # Movie C (TDKR): Matches Action (+0.03). Base sim = 0.6. Boosted sim = 0.63.
        
        understanding = QueryUnderstandingResult(
            search_intent="search",
            genres=["Action"],
            actors=["Keanu Reeves"],
            directors=["John Lasseter"]
        )

        results = asyncio.run(self.recommend_service.recommend_movies_from_understanding(understanding, limit=4))
        
        # Verify boosting applies correctly
        matrix = next(r for r in results if r["title"] == "The Matrix")
        toy_story = next(r for r in results if r["title"] == "Toy Story")
        tdkr = next(r for r in results if r["title"] == "The Dark Knight Rises")
        
        self.assertAlmostEqual(matrix["semantic_score"], 0.8, places=4)
        self.assertAlmostEqual(matrix["boosted_semantic_score"], 0.88, places=4)
        
        self.assertAlmostEqual(toy_story["semantic_score"], 0.7, places=4)
        self.assertAlmostEqual(toy_story["boosted_semantic_score"], 0.75, places=4)
        
        self.assertAlmostEqual(tdkr["semantic_score"], 0.6, places=4)
        self.assertAlmostEqual(tdkr["boosted_semantic_score"], 0.63, places=4)

        # Let's verify that the ranking reason shows the matched elements
        self.assertIn("matches preferred genre", matrix["recommendation_reason"])
        self.assertIn("Keanu Reeves", matrix["recommendation_reason"])
        self.assertIn("John Lasseter", toy_story["recommendation_reason"])

    def test_recommendation_reason_generation(self):
        """Verify dynamic recommendation reasons format correctly."""
        # Query: Horror, starring Sigourney Weaver
        # Alien (base sim = 0.9) matches both and receives boost (+0.08) -> boosted = 0.98.
        # Matrix (base sim = 0.8) matches nothing -> boosted = 0.8.
        # Alien will rank first because of the boosted semantic score.
        understanding = QueryUnderstandingResult(
            search_intent="search",
            genres=["Horror"],
            actors=["Sigourney Weaver"]
        )

        results = asyncio.run(self.recommend_service.recommend_movies_from_understanding(understanding, limit=1))
        
        self.assertEqual(results[0]["title"], "Alien")
        reason = results[0]["recommendation_reason"]
        self.assertIn("matches preferred genre(s) (Horror)", reason)
        self.assertIn("features actor(s) you like (Sigourney Weaver)", reason)

    def test_semantic_recommendation_endpoint(self):
        """Validate API route returns 200, matches schema, and applies filters end-to-end."""
        client = TestClient(app)
        
        # Mock Gemini Query Understanding to return a predefined result
        mock_understanding = QueryUnderstandingResult(
            search_intent="search",
            excluded_genres=["Comedy"],
            release_year_constraints=YearConstraints(start_year=1990)
        )
        
        # Patch QdrantClient to use our real in-memory database
        with patch("app.vector_db.qdrant.QdrantClient", return_value=self.real_memory_client):
            # Patch get_gemini_service to return mock understanding
            mock_gemini = MagicMock()
            mock_gemini.understand_query = AsyncMock(return_value=mock_understanding)
            
            # Patch get_recommendation_service to use our service
            app.dependency_overrides[get_recommendation_service] = lambda: self.recommend_service
            app.dependency_overrides[get_gemini_service] = lambda: mock_gemini
            
            try:
                response = client.get("/api/v1/recommendations/semantic?q=non-comedy%20movies%20after%201990&limit=3")
                self.assertEqual(response.status_code, 200)
                data = response.json()
                
                self.assertEqual(data["query"], "non-comedy movies after 1990")
                self.assertEqual(data["understanding"]["excluded_genres"], ["Comedy"])
                
                # Toy Story (Comedy) and Alien (< 1990) are excluded
                # Matrix and TDKR remain
                titles = [r["title"] for r in data["results"]]
                self.assertEqual(len(titles), 2)
                self.assertIn("The Matrix", titles)
                self.assertIn("The Dark Knight Rises", titles)
                self.assertNotIn("Toy Story", titles)
                self.assertNotIn("Alien", titles)
                
                # Check presence of custom reasons
                self.assertTrue(all("recommendation_reason" in r for r in data["results"]))
            finally:
                app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()
