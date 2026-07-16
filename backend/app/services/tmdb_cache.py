import sqlite3
import json
from pathlib import Path
from contextlib import closing
from typing import Optional, Dict, Any
from loguru import logger
from app.core.config import settings


class TMDbCacheManager:
    """
    Manages local SQLite cache for TMDb API responses to prevent redundant requests.
    """
    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path:
            self.db_path = Path(db_path)
        else:
            # Default to backend/app/datasets/cache/tmdb_cache.db
            self.db_path = Path(settings.PROCESSED_DATA_DIR).parent / "cache" / "tmdb_cache.db"
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a connection to the SQLite database."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        # Enable WAL mode for better concurrency during parallel writes
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        """Initializes tables for movie details and IMDb find queries."""
        logger.info(f"Initializing SQLite TMDb cache database at {self.db_path}...")
        with closing(self._get_connection()) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS movie_details_cache (
                    tmdb_id INTEGER PRIMARY KEY,
                    response_json TEXT,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS imdb_find_cache (
                    imdb_id TEXT PRIMARY KEY,
                    tmdb_id INTEGER,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS discover_cache (
                    query_hash TEXT PRIMARY KEY,
                    response_json TEXT,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()


    def _is_expired(self, fetched_at_str: str, ttl_seconds: int) -> bool:
        """Checks if a cached entry has expired based on its timestamp."""
        try:
            import datetime
            from datetime import timezone
            # SQLite CURRENT_TIMESTAMP is in UTC format: 'YYYY-MM-DD HH:MM:SS'
            dt = datetime.datetime.strptime(fetched_at_str, "%Y-%m-%d %H:%M:%S")
            dt = dt.replace(tzinfo=timezone.utc)
            elapsed = (datetime.datetime.now(timezone.utc) - dt).total_seconds()
            return elapsed > ttl_seconds
        except Exception as e:
            logger.warning(f"Error parsing cache timestamp {fetched_at_str}: {e}")
            return True

    def get_movie_details(self, tmdb_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves cached movie details response, if present and not expired."""
        try:
            with closing(self._get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT response_json, fetched_at FROM movie_details_cache WHERE tmdb_id = ?",
                    (tmdb_id,)
                )
                row = cursor.fetchone()
                if row:
                    response_json, fetched_at = row
                    if not self._is_expired(fetched_at, settings.CACHE_TTL_MOVIE_DETAILS):
                        return json.loads(response_json)
                    else:
                        logger.info(f"[TMDb Cache] Expired movie details for tmdb_id={tmdb_id}")
        except Exception as e:
            logger.warning(f"Failed to read movie details cache for tmdb_id={tmdb_id}: {e}")
        return None


    def save_movie_details(self, tmdb_id: int, response: Dict[str, Any]) -> None:
        """Saves movie details response to SQLite cache."""
        try:
            response_str = json.dumps(response, ensure_ascii=False)
            with closing(self._get_connection()) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO movie_details_cache (tmdb_id, response_json, fetched_at) VALUES (?, ?, datetime('now'))",
                    (tmdb_id, response_str)
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to save movie details cache for tmdb_id={tmdb_id}: {e}")


    def get_tmdb_id_by_imdb(self, imdb_id: str) -> Optional[int]:
        """Retrieves cached TMDb ID mapping for the given IMDb ID."""
        try:
            with closing(self._get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT tmdb_id FROM imdb_find_cache WHERE imdb_id = ?",
                    (imdb_id,)
                )
                row = cursor.fetchone()
                if row:
                    return row[0]  # could be None if unmapped
        except Exception as e:
            logger.warning(f"Failed to read IMDb mapping cache for imdb_id={imdb_id}: {e}")
        return None


    def save_imdb_mapping(self, imdb_id: str, tmdb_id: Optional[int]) -> None:
        """Saves IMDb ID to TMDb ID mapping (or None if not found) to cache."""
        try:
            with closing(self._get_connection()) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO imdb_find_cache (imdb_id, tmdb_id, fetched_at) VALUES (?, ?, datetime('now'))",
                    (imdb_id, tmdb_id)
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to save IMDb mapping cache for imdb_id={imdb_id}: {e}")

    def get_discover_results(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieves cached discover/list results, if present and not expired."""
        try:
            with closing(self._get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT response_json, fetched_at FROM discover_cache WHERE query_hash = ?",
                    (query_hash,)
                )
                row = cursor.fetchone()
                if row:
                    response_json, fetched_at = row
                    if not self._is_expired(fetched_at, settings.CACHE_TTL_DISCOVER):
                        return json.loads(response_json)
                    else:
                        logger.info(f"[TMDb Cache] Expired discover results for query_hash={query_hash}")
        except Exception as e:
            logger.warning(f"Failed to read discover cache for query_hash={query_hash}: {e}")
        return None

    def save_discover_results(self, query_hash: str, response: Dict[str, Any]) -> None:
        """Saves discover/list results to cache."""
        try:
            response_str = json.dumps(response, ensure_ascii=False)
            with closing(self._get_connection()) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO discover_cache (query_hash, response_json, fetched_at) VALUES (?, ?, datetime('now'))",
                    (query_hash, response_str)
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to save discover cache for query_hash={query_hash}: {e}")


