"""
Unit and integration tests for the ChitraAI standardized query understanding endpoint.
Validates mapping of extracted OpenAI RecommendationIntent to legacy QueryUnderstandingResult schemas.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

from fastapi.testclient import TestClient

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from main import app
from app.services.intent_extractor import IntentExtractor, RecommendationIntent, YearRange
from app.api.deps import get_intent_extractor
from app.api.routes.query import QueryUnderstandingResult, YearConstraints


class TestQueryUnderstanding(unittest.TestCase):
    """Tests for FastAPI query understanding endpoint using IntentExtractor."""

    def test_query_understand_endpoint(self):
        """Validate API route returns 200 and matches Pydantic response schema mapping."""
        client = TestClient(app)
        mock_extractor = MagicMock(spec=IntentExtractor)
        
        # Configure mock return value from OpenAI Intent Extractor
        mock_intent = RecommendationIntent(
            genres=["Horror"],
            moods=["spooky"],
            themes=["haunted house"],
            preferred_actors=["Matthew McConaughey"],
            preferred_directors=[],
            similar_movies=[],
            language="en",
            year_range=YearRange(start=2010, end=2020)
        )
        mock_extractor.extract_intent = AsyncMock(return_value=mock_intent)

        # Override dependency in FastAPI
        app.dependency_overrides[get_intent_extractor] = lambda: mock_extractor
        
        try:
            response = client.get("/api/v1/query/understand?q=spooky%20haunted%20house")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            self.assertEqual(data["search_intent"], "recommendation")
            self.assertEqual(data["mood"], "spooky")
            self.assertEqual(data["genres"], ["Horror"])
            self.assertEqual(data["themes"], ["haunted house"])
            self.assertEqual(data["actors"], ["Matthew McConaughey"])
            self.assertEqual(data["preferred_languages"], ["en"])
            self.assertIsNotNone(data["release_year_constraints"])
            self.assertEqual(data["release_year_constraints"]["start_year"], 2010)
            self.assertEqual(data["release_year_constraints"]["end_year"], 2020)
        finally:
            # Clean up overrides
            app.dependency_overrides.clear()

    def test_query_understand_validation_errors(self):
        """Verify HTTP 422 for missing query query param q."""
        client = TestClient(app)
        
        mock_extractor = MagicMock(spec=IntentExtractor)
        app.dependency_overrides[get_intent_extractor] = lambda: mock_extractor
        try:
            # Missing q query parameter
            response = client.get("/api/v1/query/understand")
            self.assertEqual(response.status_code, 422)
            
            # Empty q
            response = client.get("/api/v1/query/understand?q=")
            self.assertEqual(response.status_code, 422)
        finally:
            app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()
