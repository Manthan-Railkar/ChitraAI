from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.api.deps import qdrant_wrapper

# Import routers
from app.api.routes.health import router as health_router
from app.api.routes.search import router as search_router
from app.api.routes.recommendation import router as rec_router
from app.api.routes.movie import router as movie_router
from app.api.routes.admin import router as admin_router
from app.api.routes.query import router as query_router


# Initialize logging
setup_logging()



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Event
    logger.info("ChitraAI Backend starting up...")
    
    # Try connecting to Qdrant, but do not block startup if Qdrant is unavailable
    qdrant_connected = qdrant_wrapper.connect()
    if qdrant_connected:
        logger.info("Connected to Qdrant successfully at startup.")
    else:
        logger.warning("Could not connect to Qdrant at startup. Vector operations might fail.")
        
    yield
    
    # Shutdown Event
    logger.info("ChitraAI Backend shutting down...")

app = FastAPI(
    title="ChitraAI API",
    description="Backend services for ChitraAI semantic movie recommendations",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, configure to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routes under /api/v1
app.include_router(health_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")
app.include_router(rec_router, prefix="/api/v1")
app.include_router(movie_router, prefix="/api/v1")
app.include_router(admin_router, prefix="/api/v1")
app.include_router(query_router, prefix="/api/v1")


# Also include a top-level root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to ChitraAI Semantic Recommendation System API. Visit /docs for Swagger UI."
    }
