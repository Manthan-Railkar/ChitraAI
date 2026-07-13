from fastapi import APIRouter, Query, HTTPException, Depends
from loguru import logger

from app.services.gemini_service import GeminiService, QueryUnderstandingResult
from app.api.deps import get_gemini_service

router = APIRouter()


@router.get("/query/understand", response_model=QueryUnderstandingResult, tags=["Query Understanding"])
async def understand_query(
    q: str = Query(..., min_length=1, description="The natural language user query to analyze and extract search parameters from"),
    gemini_service: GeminiService = Depends(get_gemini_service)
) -> QueryUnderstandingResult:
    """
    Analyzes the user's natural language request to extract structured semantic parameters
    (intent, mood, themes, genres, crew, reference movies, constraints, exclusions).
    Does NOT recommend movies directly.
    """
    logger.info(f"FastAPI query understanding endpoint hit: q='{q}'")
    
    try:
        result = await gemini_service.understand_query(query=q)
        return result
    except Exception as e:
        logger.error(f"Unexpected error in FastAPI query understanding endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while analyzing the query: {str(e)}"
        )
