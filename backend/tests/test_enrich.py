import sys
import unittest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path
import sqlite3
import json
import polars as pl

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.tmdb_cache import TMDbCacheManager
from app.services.tmdb_service import TMDbService
from app.pipelines.enrich import extract_enrichment_data, enrich_single_movie, run_enrichment_pipeline
from app.core.config import settings
from app.datasets.adapters import CANONICAL_SCHEMA, CANONICAL_COLUMNS


class TestTMDbEnrichment(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # Create temp folder for caching and parquet files
        self.temp_dir = Path(__file__).resolve().parent / "temp_enrich"
        self.temp_dir.mkdir(exist_ok=True)
        (self.temp_dir / "canonical").mkdir(exist_ok=True)
        
        self.db_path = self.temp_dir / "tmdb_cache.db"
        self.parquet_path = self.temp_dir / "canonical" / "movies_knowledge_base.parquet"


        # Mock cache manager
        self.cache = TMDbCacheManager(db_path=str(self.db_path))

        # Sample mock API responses
        self.mock_details_response = {
            "id": 862,
            "title": "Toy Story",
            "overview": "Enriched overview from TMDb API.",
            "runtime": 81,
            "popularity": 45.2,
            "belongs_to_collection": {"name": "Toy Story Collection"},
            "poster_path": "/poster.jpg",
            "backdrop_path": "/backdrop.jpg",
            "genres": [{"id": 16, "name": "Animation"}, {"id": 35, "name": "Comedy"}],
            "production_companies": [{"id": 3, "name": "Pixar"}],
            "videos": {
                "results": [
                    {"site": "YouTube", "type": "Trailer", "key": "abc123trailer"}
                ]
            },
            "watch/providers": {
                "results": {
                    "US": {
                        "flatrate": [{"provider_name": "Disney Plus"}]
                    }
                }
            },
            "release_dates": {
                "results": [
                    {
                        "iso_3166_1": "US",
                        "release_dates": [{"certification": "G"}]
                    }
                ]
            },
            "keywords": {
                "keywords": [{"name": "toy"}, {"name": "friendship"}]
            }
        }

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


    def test_cache_manager(self):
        """Verifies SQLite cache manager correctly writes and reads entries."""
        # 1. Movie details caching
        self.cache.save_movie_details(999, {"title": "Test Movie"})
        cached = self.cache.get_movie_details(999)
        self.assertIsNotNone(cached)
        self.assertEqual(cached["title"], "Test Movie")

        # 2. IMDb resolved mappings
        self.cache.save_imdb_mapping("tt99999", 555)
        resolved = self.cache.get_tmdb_id_by_imdb("tt99999")
        self.assertEqual(resolved, 555)

    def test_enrichment_extraction(self):
        """Verifies that TMDb JSON response gets parsed correctly into enrichment columns."""
        parsed = extract_enrichment_data(self.mock_details_response)
        
        self.assertEqual(parsed["poster_path"], "https://image.tmdb.org/t/p/w500/poster.jpg")
        self.assertEqual(parsed["backdrop_path"], "https://image.tmdb.org/t/p/w1280/backdrop.jpg")
        self.assertEqual(parsed["collection_name"], "Toy Story Collection")
        self.assertEqual(parsed["trailer_url"], "https://www.youtube.com/watch?v={}".format("abc123trailer"))
        self.assertEqual(parsed["streaming_providers"], ["Disney Plus"])
        self.assertEqual(parsed["certification"], "G")
        self.assertEqual(parsed["runtime_minutes"], 81)
        self.assertEqual(parsed["overview"], "Enriched overview from TMDb API.")
        self.assertEqual(parsed["popularity"], 45.2)
        self.assertEqual(sorted(parsed["genres"]), ["Animation", "Comedy"])
        self.assertEqual(sorted(parsed["keywords"]), ["friendship", "toy"])
        self.assertEqual(parsed["production_companies"], ["Pixar"])

    @patch("httpx.AsyncClient.get")
    async def test_service_caching_and_api(self, mock_get):
        """Verifies service reads from cache first, otherwise hits HTTPClient and updates cache."""
        # Setup mock client response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_details_response
        mock_get.return_value = mock_response

        service = TMDbService(cache_manager=self.cache)

        # 1. First fetch: cache is empty, should trigger HTTP client request
        details = await service.fetch_movie_details(862)
        self.assertIsNotNone(details)
        self.assertEqual(details["id"], 862)
        self.assertEqual(mock_get.call_count, 1)

        # 2. Second fetch: should hit SQLite cache, NOT triggers HTTP client request again
        details_cached = await service.fetch_movie_details(862)
        self.assertIsNotNone(details_cached)
        self.assertEqual(details_cached["id"], 862)
        self.assertEqual(mock_get.call_count, 1)  # call_count remains 1

    @patch("app.services.tmdb_service.TMDbService.fetch_movie_details")
    @patch("app.services.tmdb_service.TMDbService.fetch_tmdb_id_by_imdb")
    def test_pipeline_no_overwrite(self, mock_resolve_imdb, mock_movie_details):
        """Verifies pipeline fills nulls and preserves existing valid values."""
        # Setup mock returns
        mock_resolve_imdb.return_value = None  # not needed since tmdb_id is populated
        mock_movie_details.return_value = AsyncMock()
        mock_movie_details.return_value = self.mock_details_response

        # Build mock dataset containing:
        # Row 1: movie missing overview, genres, and all enriched columns (needs enrichment)
        # Row 2: movie fully populated (genres is not null, runtime is not null). Must NOT be overwritten.
        # Row 3: movie missing tmdb_id (unlinked, skip)
        mock_data = pl.DataFrame({
            "imdb_id": ["tt0000001", "tt0000002", "tt0000003"],
            "tmdb_id": [862, 9999, None],
            "movielens_id": [1, 2, None],
            "wiki_page": [None, None, None],
            "title": ["Toy Story", "Fully Populated", "Unlinked"],
            "original_title": ["Toy Story", "Fully Populated", "Unlinked"],
            "overview": [None, "Existing Overview", "Unlinked Overview"],
            "plot_summary": [None, None, None],
            "genres": [None, ["Action", "Sci-Fi"], ["Drama"]],
            "cast": [None, None, None],
            "directors": [None, None, None],
            "writers": [None, None, None],
            "runtime_minutes": [None, 120, 90],
            "release_year": [1995, 2026, 2026],
            "rating_value": [8.0, 7.5, 5.0],
            "vote_count": [1000, 500, 100],
            "popularity": [None, 99.9, None],
            "production_companies": [None, ["Universal"], None],
            "languages": [None, None, None],
            "keywords": [None, None, None],
            "source_dataset": ["imdb", "imdb", "imdb"],
            "poster_path": [None, "/existing_poster.jpg", None],
            "backdrop_path": [None, None, None],
            "tagline": [None, None, None],
            "trailer_url": [None, None, None],
            "streaming_providers": [None, None, None],
            "collection_name": [None, None, None],
            "certification": [None, None, None],
            "document": [None, None, None]
        }).cast(CANONICAL_SCHEMA)


        # Save to temp parquet
        mock_data.write_parquet(self.parquet_path)

        # Run pipeline using mock paths in settings
        with patch("app.core.config.settings.PROCESSED_DATA_DIR", str(self.temp_dir)):
            # Set settings TMDB_API_KEY to bypass empty check
            with patch("app.core.config.settings.TMDB_API_KEY", "mock_key"):
                import asyncio
                asyncio.run(run_enrichment_pipeline(limit=10, batch_size=5))

        # Read back parquet file
        enriched = pl.read_parquet(self.parquet_path)
        self.assertEqual(enriched.height, 3)

        # Verify Row 1: Toy Story (tmdb_id=862) enriched successfully
        ts = enriched.filter(pl.col("tmdb_id") == 862)
        self.assertEqual(ts.select("poster_path").to_series().to_list()[0], "https://image.tmdb.org/t/p/w500/poster.jpg")
        self.assertEqual(ts.select("overview").to_series().to_list()[0], "Enriched overview from TMDb API.")
        self.assertEqual(ts.select("runtime_minutes").to_series().to_list()[0], 81)
        self.assertEqual(ts.select("genres").to_series().to_list()[0], ["Animation", "Comedy"])
        self.assertEqual(ts.select("certification").to_series().to_list()[0], "G")

        # Verify Row 2: Fully Populated (tmdb_id=9999) preserved its pre-existing values
        fp = enriched.filter(pl.col("tmdb_id") == 9999)
        # poster_path was "/existing_poster.jpg" and MUST NOT be overwritten by None or new value
        self.assertEqual(fp.select("poster_path").to_series().to_list()[0], "/existing_poster.jpg")
        # overview was "Existing Overview" and MUST NOT be overwritten
        self.assertEqual(fp.select("overview").to_series().to_list()[0], "Existing Overview")
        # runtime_minutes was 120 and MUST NOT be overwritten
        self.assertEqual(fp.select("runtime_minutes").to_series().to_list()[0], 120)
        # genres was ["Action", "Sci-Fi"] and MUST NOT be overwritten
        self.assertEqual(fp.select("genres").to_series().to_list()[0], ["Action", "Sci-Fi"])


if __name__ == "__main__":
    unittest.main()
