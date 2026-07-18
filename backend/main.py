from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger

from app.core.config import settings
from app.core.logging_config import setup_logging
# Import routers
from app.api.routes.health import router as health_router
from app.api.routes.search import router as search_router
from app.api.routes.recommendation import router as rec_router
from app.api.routes.movie import router as movie_router
from app.api.routes.admin import router as admin_router
from app.api.routes.query import router as query_router
from app.api.routes.stats import router as stats_router

# Initialize logging
setup_logging()

import sys
from app.core.model_manager import ModelManager
from app.api.deps import (
    local_retrieval_engine,
    embedding_service,
    recommendation_service,
    search_service,
    tmdb_service
)

def get_memory_usage_mb() -> float:
    """Returns the resident set size (RSS) memory usage of the current process in MB."""
    import os
    if os.path.exists("/proc/self/status"):
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return float(parts[1]) / 1024.0
        except Exception:
            pass
            
    try:
        import resource
        import sys
        factor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / factor
    except Exception:
        pass
        
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        pass
        
    return 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Event
    logger.info("ChitraAI Backend starting up...")
    mem_start = get_memory_usage_mb()
    logger.info(f"[Memory Profile] Initial RAM: {mem_start:.2f} MB")
    
    try:
        # Step 1, 2, 3: Defer model load
        logger.info("[Startup Step 1-3] ModelManager loading deferred (lazy loading enabled)...")
        mem_after_model = get_memory_usage_mb()
        logger.info(f"[Memory Profile] RAM after model deferral: {mem_after_model:.2f} MB (Change: {mem_after_model - mem_start:+.2f} MB)")
        
        # Step 4: Initialize Local Retrieval Engine (loads parquet metadata only, connects to Qdrant without syncing)
        logger.info("[Startup Step 4] Initializing Local Retrieval Engine...")
        local_retrieval_engine.initialize()
        mem_after_retrieval = get_memory_usage_mb()
        logger.info(f"[Memory Profile] RAM after Local Retrieval Engine init: {mem_after_retrieval:.2f} MB (Change: {mem_after_retrieval - mem_after_model:+.2f} MB)")
        
        # Step 5: Register all services
        logger.info("[Startup Step 5] Registering all service dependencies...")
        logger.info("Services registered: LocalRetrievalEngine, EmbeddingService, SearchService, RecommendationService, TMDbService.")
        
        # Step 6: Start accepting requests
        logger.info("[Startup Step 6] Backend initialization completed successfully. Server is ready to accept requests.")
        mem_final = get_memory_usage_mb()
        logger.info(f"[Memory Profile] Final Startup RAM: {mem_final:.2f} MB")
        
    except Exception as e:
        logger.critical(f"ChitraAI Backend startup aborted due to critical initialization failure: {e}")
        sys.exit(1)
        
    yield
    
    # Shutdown Event
    logger.info("ChitraAI Backend shutting down...")
    try:
        await tmdb_service.close()
    except Exception as e:
        logger.error(f"Error closing TMDbService persistent client: {e}")


app = FastAPI(
    title="ChitraAI API",
    description="Backend services for ChitraAI semantic movie recommendations",
    version="1.0.0",
    lifespan=lifespan
)

from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request, exc: StarletteHTTPException):
    code_map = {
        404: "NOT_FOUND",
        400: "BAD_REQUEST",
        422: "VALIDATION_ERROR",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        500: "INTERNAL_SERVER_ERROR"
    }
    code = code_map.get(exc.status_code, "ERROR")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": code,
                "message": exc.detail
            }
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    errors = exc.errors()
    if errors:
        msg = f"Validation failed: {errors[0].get('loc')} - {errors[0].get('msg')}"
    else:
        msg = "Validation failed"
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": msg
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    logger.error(f"Unhandled exception occurred: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected internal server error occurred."
            }
        }
    )

# CORS Configuration
allowed_origins = [origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()]
for local_origin in ["http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://127.0.0.1:5173", "http://127.0.0.1:5174", "http://127.0.0.1:5175"]:
    if local_origin not in allowed_origins:
        allowed_origins.append(local_origin)
if not allowed_origins or "*" in allowed_origins:
    allowed_origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routes under /api/v1 and /api
for prefix in ["/api/v1", "/api"]:
    app.include_router(health_router, prefix=prefix)
    app.include_router(search_router, prefix=prefix)
    app.include_router(rec_router, prefix=prefix)
    app.include_router(movie_router, prefix=prefix)
    app.include_router(admin_router, prefix=prefix)
    app.include_router(query_router, prefix=prefix)
    app.include_router(stats_router, prefix=prefix)


# Also include a top-level root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to ChitraAI Semantic Recommendation System API. Visit /docs for Swagger UI."
    }
