"""
Unit and integration tests for the ChitraAI Gemini Query Understanding module.
Validates structured JSON parsing, backoff retries, local heuristics fallback,
in-memory caching, and FastAPI route responses.
"""

import sys
import unittest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import httpx
from fastapi.testclient import TestClient

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from main import app
from app.services.gemini_service import GeminiService, QueryUnderstandingResult, YearConstraints


class TestGeminiQueryUnderstanding(unittest.TestCase):
    """Tests for GeminiService and FastAPI query understanding endpoint."""

    def setUp(self):
        # Initialize GeminiService with mock key
        self.service = GeminiService(api_key="mock_gemini_api_key", model="gemini-1.5-flash")

    def test_heuristic_extraction_parsing(self):
        """Verify the local heuristic fallback extractor regex parsing logic."""
        # 1. Test genre & mood & themes inclusion
        q1 = "I want a scary sci-fi movie about space exploration"
        res1 = self.service._extract_heuristics(q1)
        self.assertEqual(res1.search_intent, "search")
        self.assertEqual(res1.mood, "scary")
        self.assertIn("Sci-Fi", res1.genres)
        self.assertIn("Horror", res1.genres)
        self.assertIn("space exploration", res1.themes)

        # 2. Test year constraints: after / before / in
        q2 = "romantic comedy released after 1995 but before 2010"
        res2 = self.service._extract_heuristics(q2)
        self.assertIsNotNone(res2.release_year_constraints)
        self.assertEqual(res2.release_year_constraints.start_year, 1996)
        self.assertEqual(res2.release_year_constraints.end_year, 2009)

        q3 = "action movie in 2012"
        res3 = self.service._extract_heuristics(q3)
        self.assertIsNotNone(res3.release_year_constraints)
        self.assertEqual(res3.release_year_constraints.exact_year, 2012)

        # 3. Test genre exclusion (e.g. no comedy, without horror)
        q4 = "thriller but without comedy and not horror"
        res4 = self.service._extract_heuristics(q4)
        self.assertIn("Comedy", res4.excluded_genres)
        self.assertIn("Horror", res4.excluded_genres)

    @patch("httpx.AsyncClient.post")
    def test_gemini_api_success(self, mock_post):
        """Verify successful Gemini API response is correctly validated and parsed."""
        query = "space movie with Matthew McConaughey"
        
        # Predefined mock JSON response from Gemini API
        mock_response_body = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": '{"search_intent": "search", "mood": "epic", "themes": ["space travel"], "genres": ["Sci-Fi", "Drama"], "actors": ["Matthew McConaughey"], "directors": [], "reference_movies": [], "preferred_languages": ["en"], "release_year_constraints": null, "excluded_genres": [], "user_preferences": null}'
                    }]
                }
            }]
        }

        # Mock the HTTP response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_body
        mock_post.return_value = mock_response

        # Invoke service
        res = asyncio.run(self.service.understand_query(query))
        
        self.assertTrue(mock_post.called)
        self.assertEqual(res.search_intent, "search")
        self.assertEqual(res.mood, "epic")
        self.assertIn("Sci-Fi", res.genres)
        self.assertIn("Matthew McConaughey", res.actors)

    @patch("httpx.AsyncClient.post")
    def test_gemini_api_retries(self, mock_post):
        """Verify backoff retry logic triggers on 429/503 errors and eventually succeeds."""
        query = "spooky movie"
        
        mock_response_body = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": '{"search_intent": "search", "mood": "spooky", "genres": ["Horror"]}'
                    }]
                }
            }]
        }

        # Mock responses: 429 -> 503 -> 200
        mock_response_429 = MagicMock(spec=httpx.Response)
        mock_response_429.status_code = 429
        mock_response_429.text = "Rate Limit"

        mock_response_503 = MagicMock(spec=httpx.Response)
        mock_response_503.status_code = 503
        mock_response_503.text = "Unavailable"

        mock_response_200 = MagicMock(spec=httpx.Response)
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = mock_response_body

        mock_post.side_effect = [mock_response_429, mock_response_503, mock_response_200]

        # Patch asyncio.sleep to avoid waiting during test executions
        with patch("asyncio.sleep", return_value=None) as mock_sleep:
            res = asyncio.run(self.service.understand_query(query))
            
            self.assertEqual(mock_sleep.call_count, 2)
            self.assertEqual(res.search_intent, "search")
            self.assertEqual(res.mood, "spooky")
            self.assertIn("Horror", res.genres)

    @patch("httpx.AsyncClient.post")
    def test_gemini_cache(self, mock_post):
        """Verify that duplicate queries hit the local in-memory cache and skip API calls."""
        query = "funny cartoon"
        
        mock_response_body = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": '{"search_intent": "search", "genres": ["Animation", "Comedy"]}'
                    }]
                }
            }]
        }

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_body
        mock_post.return_value = mock_response

        # First call: hits API
        res1 = asyncio.run(self.service.understand_query(query))
        self.assertEqual(mock_post.call_count, 1)

        # Second call: hits cache, doesn't call API again
        res2 = asyncio.run(self.service.understand_query(query))
        self.assertEqual(mock_post.call_count, 1)
        
        self.assertEqual(res1.genres, res2.genres)

    @patch("httpx.AsyncClient.post")
    def test_gemini_api_failures_fallback(self, mock_post):
        """Verify persistent API errors fallback to heuristic extraction instead of failing."""
        query = "scary space movie released after 2010"

        # Mock responses: consistently returning 500
        mock_response_500 = MagicMock(spec=httpx.Response)
        mock_response_500.status_code = 500
        mock_response_500.text = "Server Error"
        mock_post.return_value = mock_response_500

        with patch("asyncio.sleep", return_value=None):
            # Should not throw exception; should fallback to heuristics
            res = asyncio.run(self.service.understand_query(query))
            
            self.assertEqual(res.search_intent, "search")
            self.assertEqual(res.mood, "scary")
            self.assertIn("Horror", res.genres)
            self.assertIn("space exploration", res.themes)
            self.assertEqual(res.release_year_constraints.start_year, 2011)

    def test_query_understand_endpoint(self):
        """Validate API route returns 200 and matches Pydantic response schema."""
        client = TestClient(app)
        mock_service = MagicMock(spec=GeminiService)
        
        # Configure mock return value
        mock_result = QueryUnderstandingResult(
            search_intent="search",
            mood="spooky",
            genres=["Horror"],
            themes=["haunted house"]
        )
        # Mock async method return value using AsyncMock
        mock_service.understand_query = AsyncMock(return_value=mock_result)

        # Override dependency in FastAPI
        from app.api.deps import get_gemini_service
        app.dependency_overrides[get_gemini_service] = lambda: mock_service
        
        try:
            response = client.get("/api/v1/query/understand?q=spooky%20haunted%20house")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            self.assertEqual(data["search_intent"], "search")
            self.assertEqual(data["mood"], "spooky")
            self.assertEqual(data["genres"], ["Horror"])
            self.assertEqual(data["themes"], ["haunted house"])
        finally:
            # Clean up overrides
            app.dependency_overrides.clear()


    def test_query_understand_validation_errors(self):
        """Verify HTTP 422 for missing query query param q."""
        client = TestClient(app)
        
        # Missing q query parameter
        response = client.get("/api/v1/query/understand")
        self.assertEqual(response.status_code, 422)
        
        # Empty q
        response = client.get("/api/v1/query/understand?q=")
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
