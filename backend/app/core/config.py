import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory of the backend folder
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    TMDB_API_KEY: str = ""
    OMDB_API_KEY: str = ""
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "movies"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    DEVICE: str = "cpu"
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: str = ""

    # Dataset Directories
    RAW_DATA_DIR: str = str(BASE_DIR / "app" / "datasets" / "raw")
    PROCESSED_DATA_DIR: str = str(BASE_DIR / "app" / "datasets" / "processed")
    MERGED_DATA_DIR: str = str(BASE_DIR / "app" / "datasets" / "merged")
    REPORTS_DIR: str = str(BASE_DIR / "reports")

    # Embedding Pipeline Settings
    EMBEDDINGS_DIR: str = str(BASE_DIR / "app" / "embeddings")
    EMBEDDING_BATCH_SIZE: int = 256
    EMBEDDING_CHECKPOINT_INTERVAL: int = 50

    # Qdrant Ingestion Settings
    QDRANT_PATH: str | None = str(BASE_DIR / "app" / "vector_db" / "qdrant_local")
    QDRANT_BATCH_SIZE: int = 500
    QDRANT_INGEST_CHECKPOINT_INTERVAL: int = 10

    # LLM Provider API Settings
    GROQ_API_KEY: str = ""
    LLM_PROVIDER: str = "groq"
    LLM_MODEL: str = "llama-3.1-8b-instant"

    # Recommendation Fallback Settings
    SEMANTIC_CONFIDENCE_THRESHOLD: float = 0.70

    # Hybrid Retrieval Engine Settings
    HYBRID_SEMANTIC_WEIGHT: float = 0.55
    HYBRID_BM25_WEIGHT: float = 0.25
    HYBRID_METADATA_WEIGHT: float = 0.20

    # Feature Flags
    USE_TMDB_RETRIEVAL: bool = True
    METRICS_ENABLED: bool = True

    # --- Cache TTLs (seconds) ---
    CACHE_TTL_MOVIE_DETAILS: int = 86400      # 24 hours
    CACHE_TTL_DISCOVER: int = 3600             # 1 hour
    CACHE_TTL_SEARCH: int = 3600               # 1 hour
    CACHE_TTL_INTENT: int = 1800               # 30 minutes

    # --- Cache Sizes ---
    RESPONSE_CACHE_SIZE: int = 200
    INTENT_CACHE_SIZE: int = 200

    # --- TMDb API Settings ---
    TMDB_TIMEOUT: float = 8.0
    TMDB_MAX_RETRIES: int = 3
    TMDB_POOL_SIZE: int = 20
    TMDB_SEMAPHORE_LIMIT: int = 10
    TMDB_BACKOFF_BASE: float = 0.5
    TMDB_BACKOFF_MAX: float = 8.0

    # --- Groq / LLM API Settings ---
    GROQ_TIMEOUT: float = 15.0
    GROQ_MAX_RETRIES: int = 2

    # --- General Performance ---
    FUSION_CANDIDATES_LIMIT: int = 1000

    # Legacy aliases (kept for backward compatibility)
    CACHE_SIZE: int = 200
    TIMEOUT_SECONDS: float = 8.0
    MAX_RETRIES: int = 3


# Instantiate settings to be imported globally
settings = Settings()
