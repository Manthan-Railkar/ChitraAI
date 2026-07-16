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
from app.services.embedding_service import EmbeddingService
from app.services.recommendation_service import RecommendationService
from app.services.local_retrieval import LocalRetrievalEngine
from app.services.intent_extractor import IntentExtractor, RecommendationIntent
from app.api.routes.query import QueryUnderstandingResult, YearConstraints
from app.api.deps import get_recommendation_service
import polars as pl


class TestHybridRecommendation(unittest.TestCase):
    """Tests for RecommendationService and FastAPI recommendations route."""

    def setUp(self):
        # 1. Setup mock embedding service
        self.mock_embedding_service = MagicMock(spec=EmbeddingService)
        # Mock 768-dim query embedding vector
        self.mock_query_vector = np.zeros(768, dtype=np.float32)
        self.mock_query_vector[0] = 1.0  # simple indicator vector
        self.mock_embedding_service.encode_single.return_value = self.mock_query_vector

        # 2. Setup mock vectors
        vec_alien = np.zeros(768, dtype=np.float32)
        vec_alien[0] = 0.9
        vec_alien[1] = 0.43589
        
        vec_matrix = np.zeros(768, dtype=np.float32)
        vec_matrix[0] = 0.8
        vec_matrix[1] = 0.6
        
        vec_toy_story = np.zeros(768, dtype=np.float32)
        vec_toy_story[0] = 0.7
        vec_toy_story[1] = 0.71414
        
        vec_tdkr = np.zeros(768, dtype=np.float32)
        vec_tdkr[0] = 0.6
        vec_tdkr[1] = 0.8

        # 3. Create LocalRetrievalEngine with mock details
        self.local_engine = LocalRetrievalEngine(self.mock_embedding_service)
        self.local_engine.movies_df = pl.DataFrame([
            {
                "tmdb_id": 1,
                "imdb_id": "tt0133093",
                "movielens_id": 1,
                "wiki_page": "https://en.wikipedia.org/wiki/The_Matrix",
                "title": "The Matrix",
                "original_title": "The Matrix",
                "overview": "A computer hacker learns from mysterious rebels about the true nature of his reality.",
                "plot_summary": None,
                "genres": ["Action", "Sci-Fi"],
                "cast": ["Keanu Reeves", "Laurence Fishburne"],
                "directors": ["Lana Wachowski", "Lilly Wachowski"],
                "writers": ["Lana Wachowski", "Lilly Wachowski"],
                "runtime_minutes": 136,
                "release_year": 1999,
                "rating_value": 8.7,
                "vote_count": 22000,
                "popularity": 80.0,
                "production_companies": [],
                "languages": ["en"],
                "keywords": [],
                "source_dataset": "tmdb",
                "poster_path": "/matrix.jpg",
                "backdrop_path": None,
                "trailer_url": None,
                "streaming_providers": [],
                "collection_name": None,
                "certification": None,
                "tagline": "Welcome to the Real World",
                "document": ""
            },
            {
                "tmdb_id": 2,
                "imdb_id": "tt0114709",
                "movielens_id": 2,
                "wiki_page": "https://en.wikipedia.org/wiki/Toy_Story",
                "title": "Toy Story",
                "original_title": "Toy Story",
                "overview": "A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy's room.",
                "plot_summary": None,
                "genres": ["Animation", "Comedy"],
                "cast": ["Tom Hanks", "Tim Allen"],
                "directors": ["John Lasseter"],
                "writers": [],
                "runtime_minutes": 81,
                "release_year": 1995,
                "rating_value": 8.3,
                "vote_count": 15000,
                "popularity": 60.0,
                "production_companies": [],
                "languages": ["en"],
                "keywords": [],
                "source_dataset": "tmdb",
                "poster_path": "/toystory.jpg",
                "backdrop_path": None,
                "trailer_url": None,
                "streaming_providers": [],
                "collection_name": None,
                "certification": None,
                "tagline": "The toys are back in town",
                "document": ""
            },
            {
                "tmdb_id": 3,
                "imdb_id": "tt1345836",
                "movielens_id": 3,
                "wiki_page": "https://en.wikipedia.org/wiki/The_Dark_Knight_Rises",
                "title": "The Dark Knight Rises",
                "original_title": "The Dark Knight Rises",
                "overview": "Eight years after the Joker's reign of anarchy, Batman, with the help of the enigmatic Catwoman, is forced from his exile to save Gotham City from the brutal guerrilla terrorist Bane.",
                "plot_summary": None,
                "genres": ["Action", "Thriller"],
                "cast": ["Christian Bale", "Gary Oldman"],
                "directors": ["Christopher Nolan"],
                "writers": [],
                "runtime_minutes": 165,
                "release_year": 2012,
                "rating_value": 8.4,
                "vote_count": 25000,
                "popularity": 90.0,
                "production_companies": [],
                "languages": ["en"],
                "keywords": [],
                "source_dataset": "tmdb",
                "poster_path": "/tdkr.jpg",
                "backdrop_path": None,
                "trailer_url": None,
                "streaming_providers": [],
                "collection_name": None,
                "certification": None,
                "tagline": "A Legend Ends",
                "document": ""
            },
            {
                "tmdb_id": 4,
                "imdb_id": "tt0078748",
                "movielens_id": 4,
                "wiki_page": "https://en.wikipedia.org/wiki/Alien_(film)",
                "title": "Alien",
                "original_title": "Alien",
                "overview": "After a space merchant vessel receives an unknown transmission as a distress call, one of the crew is attacked by a mysterious lifeform and its journey to Earth is interrupted.",
                "plot_summary": None,
                "genres": ["Horror", "Sci-Fi"],
                "cast": ["Sigourney Weaver", "Tom Skerritt"],
                "directors": ["Ridley Scott"],
                "writers": [],
                "runtime_minutes": 117,
                "release_year": 1979,
                "rating_value": 8.5,
                "vote_count": 9000,
                "popularity": 45.0,
                "production_companies": [],
                "languages": ["en"],
                "keywords": [],
                "source_dataset": "tmdb",
                "poster_path": "/alien.jpg",
                "backdrop_path": None,
                "trailer_url": None,
                "streaming_providers": [],
                "collection_name": None,
                "certification": None,
                "tagline": "In space no one can hear you scream",
                "document": ""
            }
        ])

        # Convert to matrix
        self.local_engine.embeddings_matrix = np.array([vec_matrix, vec_toy_story, vec_tdkr, vec_alien], dtype=np.float32)
        norms = np.linalg.norm(self.local_engine.embeddings_matrix, axis=1, keepdims=True)
        self.local_engine.embeddings_matrix = self.local_engine.embeddings_matrix / np.where(norms == 0, 1e-12, norms)
        
        self.local_engine.tmdb_id_to_idx = {
            1: 0,
            2: 1,
            3: 2,
            4: 3
        }

        # 4. Initialize RecommendationService under test
        self.mock_intent_extractor = MagicMock(spec=IntentExtractor)
        self.recommend_service = RecommendationService(self.local_engine, self.mock_intent_extractor)

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
        
        # Setup mock intent extractor to return the expected intent for the query
        from app.services.intent_extractor import YearRange
        self.mock_intent_extractor.extract_intent = AsyncMock(
            return_value=RecommendationIntent(
                avoid_genres=["Comedy"],
                year_range=YearRange(start=1990)
            )
        )
        
        # Patch get_recommendation_service to use our service
        app.dependency_overrides[get_recommendation_service] = lambda: self.recommend_service
        
        try:
            response = client.get("/api/v1/recommendations/semantic?q=non-comedy%20movies%20after%201990&limit=3")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            self.assertEqual(data["query"], "non-comedy movies after 1990")
            self.assertEqual(data["metadata"]["understanding"]["excluded_genres"], ["Comedy"])
            
            # Toy Story (Comedy) and Alien (< 1990) are excluded
            # Matrix and TDKR remain
            titles = [r["title"] for r in data["recommendations"]]
            self.assertEqual(len(titles), 2)
            self.assertIn("The Matrix", titles)
            self.assertIn("The Dark Knight Rises", titles)
            self.assertNotIn("Toy Story", titles)
            self.assertNotIn("Alien", titles)
            
            # Check presence of custom reasons
            self.assertTrue(all("recommendation_reason" in r for r in data["recommendations"]))
        finally:
            app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()
