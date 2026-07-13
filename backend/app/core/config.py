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
    QDRANT_COLLECTION: str = "movies"
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    DEVICE: str = "cpu"
    LOG_LEVEL: str = "INFO"

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

    # Gemini API Settings
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"




# Instantiate settings to be imported globally
settings = Settings()

