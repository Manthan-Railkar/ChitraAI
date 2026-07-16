from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, Query, HTTPException, Depends
from loguru import logger

from app.services.intent_extractor import IntentExtractor, RecommendationIntent
from app.api.deps import get_intent_extractor

router = APIRouter()


class YearConstraints(BaseModel):
    """Schemas representing release year constraints."""
    start_year: Optional[int] = Field(None, description="Minimum release year constraint (inclusive)")
    end_year: Optional[int] = Field(None, description="Maximum release year constraint (inclusive)")
    exact_year: Optional[int] = Field(None, description="Exact release year constraint")


class QueryUnderstandingResult(BaseModel):
    """Schema representing structured query understanding analysis output."""
    search_intent: str = Field(default="unknown", description="Primary query intent (search, recommendation, comparison, etc.)")
    mood: Optional[str] = Field(None, description="Mood or tone extracted from query")
    themes: List[str] = Field(default_factory=list, description="Extracted themes or plot elements")
    genres: List[str] = Field(default_factory=list, description="Target genres matching dataset standards")
    actors: List[str] = Field(default_factory=list, description="Actor names mentioned")
    directors: List[str] = Field(default_factory=list, description="Director names mentioned")
    reference_movies: List[str] = Field(default_factory=list, description="Specific movies referenced in query")
    preferred_languages: List[str] = Field(default_factory=list, description="Languages preferred")
    release_year_constraints: Optional[YearConstraints] = Field(None, description="Extracted release year boundaries")
    excluded_genres: List[str] = Field(default_factory=list, description="Genres explicitly excluded")
    user_preferences: Optional[str] = Field(None, description="Additional custom user requests or constraints")
    
    # Extended router fields
    intent: Optional[str] = Field(None, description="The classified search intent (movie_lookup, recommendation, general_search)")
    ranking_mode: Optional[str] = Field(None, description="The selected ranking mode (best, similar_movie, mood, default)")
    release_year: Optional[int] = Field(None, description="Exact release year extracted")
    exclusions: List[str] = Field(default_factory=list, description="Extracted generic exclusions")


def intent_to_query_understanding_result(intent: RecommendationIntent) -> QueryUnderstandingResult:
    """Helper to convert OpenAI RecommendationIntent to legacy QueryUnderstandingResult schema."""
    year_constraints = None
    if intent.year_range:
        exact = None
        if intent.year_range.start and intent.year_range.start == intent.year_range.end:
            exact = intent.year_range.start
        year_constraints = YearConstraints(
            start_year=intent.year_range.start,
            end_year=intent.year_range.end,
            exact_year=exact
        )

    mood = ", ".join(intent.moods) if intent.moods else None
    preferred_languages = [intent.language] if intent.language else []

    # Map search_intent: if we have any search parameters, treat as recommendation
    raw_intent = getattr(intent, "intent", "recommendation")
    if not raw_intent or raw_intent == "unknown":
        has_params = bool(
            intent.genres or intent.moods or intent.themes or
            intent.preferred_actors or intent.preferred_directors or
            intent.similar_movies
        )
        search_intent = "recommendation" if has_params else "search"
    else:
        search_intent = raw_intent

    return QueryUnderstandingResult(
        search_intent=search_intent,
        mood=mood,
        themes=intent.themes,
        genres=intent.genres,
        actors=intent.preferred_actors,
        directors=intent.preferred_directors,
        reference_movies=intent.similar_movies,
        preferred_languages=preferred_languages,
        release_year_constraints=year_constraints,
        excluded_genres=intent.avoid_genres,
        user_preferences=None,
        intent=search_intent,
        ranking_mode=getattr(intent, "ranking_mode", "default"),
        release_year=getattr(intent, "release_year", None),
        exclusions=getattr(intent, "exclusions", [])
    )


@router.get("/query/understand", response_model=QueryUnderstandingResult, tags=["Query Understanding"])
async def understand_query(
    q: str = Query(..., min_length=1, description="The natural language user query to analyze and extract search parameters from"),
    intent_extractor: IntentExtractor = Depends(get_intent_extractor)
) -> QueryUnderstandingResult:
    """
    Analyzes the user's natural language request to extract structured semantic parameters
    using OpenAI's intent extractor service.
    """
    logger.info(f"FastAPI query understanding endpoint hit (OpenAI): q='{q}'")
    
    try:
        intent = await intent_extractor.extract_intent(query=q)
        return intent_to_query_understanding_result(intent)
    except Exception as e:
        logger.error(f"Unexpected error in query understanding endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while analyzing the query: {str(e)}"
        )
