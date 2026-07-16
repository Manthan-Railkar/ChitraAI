import pytest
from app.services.intent_extractor import RecommendationIntent, YearRange
from app.services.tmdb_query_builder import TMDbQueryBuilder
from app.services.tmdb_service import TMDbService
from app.services.recommendation_service import RecommendationService
from app.core.config import settings

class DummyTMDbService:
    def __init__(self):
        self.api_key = "dummy_key"
        
    async def resolve_keyword_id(self, keyword: str):
        return 1234
        
    async def resolve_person_id(self, name: str):
        return 5678

@pytest.mark.anyio
async def test_tmdb_query_builder_genre_and_language():
    intent = RecommendationIntent(
        genres=["Action", "Science Fiction"],
        language="Korean",
        release_year=2021,
        runtime=120
    )
    dummy_service = DummyTMDbService()
    
    params = await TMDbQueryBuilder.build_query(intent, dummy_service)
    
    assert params["with_genres"] == "28,878"
    assert params["with_original_language"] == "ko"
    assert params["primary_release_year"] == 2021
    assert params["with_runtime.lte"] == 120

@pytest.mark.anyio
async def test_tmdb_query_builder_year_range():
    intent = RecommendationIntent(
        year_range=YearRange(start=2010, end=2015)
    )
    dummy_service = DummyTMDbService()
    
    params = await TMDbQueryBuilder.build_query(intent, dummy_service)
    
    assert params["primary_release_date.gte"] == "2010-01-01"
    assert params["primary_release_date.lte"] == "2015-12-31"

@pytest.mark.anyio
async def test_tmdb_query_builder_exclusions():
    intent = RecommendationIntent(
        avoid_genres=["Comedy", "Horror"]
    )
    dummy_service = DummyTMDbService()
    
    params = await TMDbQueryBuilder.build_query(intent, dummy_service)
    
    assert params["without_genres"] == "35,27"
