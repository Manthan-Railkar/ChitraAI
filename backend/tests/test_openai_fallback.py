import sys
import asyncio
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from main import app
from app.services.openai_service import OpenAIService
from app.api.routes.recommendation import evaluate_semantic_confidence, recommendation_cache
from app.core.config import settings


class TestOpenAIFallback(unittest.TestCase):
    def setUp(self):
        # Reset request cache
        recommendation_cache.cache.clear()

    def test_evaluate_semantic_confidence(self):
        # Empty results should return 0.0 confidence
        self.assertEqual(evaluate_semantic_confidence([], 10), 0.0)

        # High similarity score, full metadata, count matches limit
        good_results = [
            {"semantic_score": 0.85, "title": "Inception", "overview": "Dreams", "genres": ["Sci-Fi"]},
            {"semantic_score": 0.82, "title": "Matrix", "overview": "Virtual reality", "genres": ["Action", "Sci-Fi"]}
        ]
        conf = evaluate_semantic_confidence(good_results, 2)
        # Expected computation:
        # top_score (0.85) * 0.7 = 0.595
        # count_ratio (2/2) * 0.15 = 0.15
        # completeness (3/3) * 0.15 = 0.15
        # Total = 0.895
        self.assertAlmostEqual(conf, 0.895, places=4)

    @patch("app.core.model_manager.ModelManager.get_openai_client")
    def test_openai_service_fallback(self, mock_get_client):
        # Mock OpenAI completions client
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = (
            '{"results": [{'
            '  "title": "Interstellar",'
            '  "release_year": 2014,'
            '  "genres": ["Sci-Fi", "Adventure"],'
            '  "overview": "A team of explorers travel through a wormhole...",'
            '  "rating_value": 8.6,'
            '  "popularity": 95.0,'
            '  "vote_count": 15000,'
            '  "recommendation_reason": "Matches space travel theme."'
            '}]}'
        )
        mock_choice.message = mock_message
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
        mock_get_client.return_value = mock_client

        service = OpenAIService()
        res = asyncio.run(service.get_fallback_recommendations("space travel", 1))
        
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["title"], "Interstellar")
        self.assertEqual(res[0]["release_year"], 2014)
        self.assertEqual(res[0]["rating_value"], 8.6)
        self.assertEqual(res[0]["recommendation_reason"], "Matches space travel theme.")


if __name__ == "__main__":
    unittest.main()
