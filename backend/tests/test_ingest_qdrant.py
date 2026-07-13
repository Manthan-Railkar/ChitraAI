"""
Unit and integration tests for the ChitraAI Qdrant Ingestion Pipeline.
Uses a real in-memory Qdrant client to test collection creation, deduplication,
batch uploading, checkpointing, and retry logic.
"""

import sys
import json
import uuid
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import polars as pl
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.datasets.adapters import CANONICAL_SCHEMA
from app.vector_db.qdrant import QdrantWrapper
from app.pipelines.ingest_qdrant import (
    generate_movie_uuid,
    ingest_movies_to_qdrant
)


def _make_test_metadata(n_rows: int = 20) -> pl.DataFrame:
    """Creates a minimal test metadata knowledge base DataFrame."""
    data = {
        "imdb_id": [f"tt{1000000 + i}" for i in range(n_rows)],
        "tmdb_id": list(range(1, n_rows + 1)),
        "movielens_id": list(range(1, n_rows + 1)),
        "wiki_page": [None] * n_rows,
        "title": [f"Test Movie {i}" for i in range(n_rows)],
        "original_title": [f"Test Movie {i}" for i in range(n_rows)],
        "overview": [f"Overview {i}" for i in range(n_rows)],
        "plot_summary": [None] * n_rows,
        "genres": [["Drama"] for _ in range(n_rows)],
        "cast": [["Actor A"] for _ in range(n_rows)],
        "directors": [["Director A"] for _ in range(n_rows)],
        "writers": [None] * n_rows,
        "runtime_minutes": [120] * n_rows,
        "release_year": [2000 + i for i in range(n_rows)],
        "rating_value": [7.5] * n_rows,
        "vote_count": [1000] * n_rows,
        "popularity": [50.0] * n_rows,
        "production_companies": [None] * n_rows,
        "languages": [["en"]] * n_rows,
        "keywords": [None] * n_rows,
        "source_dataset": ["imdb"] * n_rows,
        "poster_path": [None] * n_rows,
        "backdrop_path": [None] * n_rows,
        "trailer_url": [None] * n_rows,
        "streaming_providers": [None] * n_rows,
        "collection_name": [None] * n_rows,
        "certification": [None] * n_rows,
        "document": [f"Title: Test Movie {i} ({2000 + i})\nGenres: Drama" for i in range(n_rows)],
    }
    return pl.DataFrame(data).cast(CANONICAL_SCHEMA)


def _make_test_embeddings(n_rows: int = 20, dim: int = 4) -> pl.DataFrame:
    """Creates a minimal test embeddings DataFrame."""
    embeddings_list = [[float(i)] * dim for i in range(n_rows)]
    data = {
        "imdb_id": [f"tt{1000000 + i}" for i in range(n_rows)],
        "tmdb_id": list(range(1, n_rows + 1)),
        "movielens_id": list(range(1, n_rows + 1)),
        "title": [f"Test Movie {i}" for i in range(n_rows)],
        "release_year": [2000 + i for i in range(n_rows)],
        "embedding": embeddings_list
    }
    return pl.DataFrame(data)


class TestQdrantIngestPipeline(unittest.TestCase):
    """Tests for Qdrant Ingestion Wrapper and Pipeline Orchestrator."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.processed_dir = Path(self.tmpdir) / "processed" / "canonical"
        self.processed_dir.mkdir(parents=True)
        self.merged_dir = Path(self.tmpdir) / "merged"
        self.merged_dir.mkdir(parents=True)
        self.embeddings_dir = Path(self.tmpdir) / "embeddings"
        self.embeddings_dir.mkdir(parents=True)
        self.reports_dir = Path(self.tmpdir) / "reports"
        self.reports_dir.mkdir(parents=True)

        # Write test kb and embeddings Parquet files
        self.metadata_df = _make_test_metadata(n_rows=20)
        self.metadata_df.write_parquet(self.merged_dir / "movies_knowledge_base.parquet")

        self.embeddings_df = _make_test_embeddings(n_rows=20, dim=4)
        self.embeddings_df.write_parquet(self.embeddings_dir / "movie_embeddings.parquet")

        # In-memory Qdrant client shared for tests
        self.real_memory_client = QdrantClient(location=":memory:")

        # Setup mock settings patcher
        self.mock_settings = MagicMock()
        self.mock_settings.PROCESSED_DATA_DIR = str(Path(self.tmpdir) / "processed")
        self.mock_settings.MERGED_DATA_DIR = str(self.merged_dir)
        self.mock_settings.EMBEDDINGS_DIR = str(self.embeddings_dir)
        self.mock_settings.REPORTS_DIR = str(self.reports_dir)
        self.mock_settings.QDRANT_URL = "mock-url"
        self.mock_settings.QDRANT_PATH = str(Path(self.tmpdir) / "qdrant_local")
        self.mock_settings.QDRANT_COLLECTION = "movies_test"
        self.mock_settings.QDRANT_BATCH_SIZE = 5
        self.mock_settings.QDRANT_INGEST_CHECKPOINT_INTERVAL = 2

    def _patch_qdrant_client(self):
        """Patcher to force QdrantWrapper to use our real in-memory QdrantClient."""
        return patch("app.vector_db.qdrant.QdrantClient", return_value=self.real_memory_client)

    def _patch_settings(self):
        """Patcher for configurations."""
        return patch("app.pipelines.ingest_qdrant.settings", self.mock_settings)

    def test_qdrant_wrapper_connect_memory(self):
        """Verify QdrantWrapper successfully connects using the mock/memory target."""
        with self._patch_qdrant_client():
            wrapper = QdrantWrapper(collection_name="movies_test")
            self.assertTrue(wrapper.connect())
            self.assertFalse(wrapper._is_local) # Remote is mocked but points to memory client

    def test_deterministic_uuid_generation(self):
        """Verify generated UUIDs are deterministic and prevent duplicates."""
        uuid1 = generate_movie_uuid("tt1234567", 550, "Fight Club", 1999)
        uuid2 = generate_movie_uuid("tt1234567", 550, "Fight Club", 1999)
        uuid3 = generate_movie_uuid("tt0000000", 550, "Fight Club", 1999)
        
        self.assertEqual(uuid1, uuid2)
        self.assertNotEqual(uuid1, uuid3)

        # Empty IDs fallback
        uuid_fallback1 = generate_movie_uuid(None, None, "A Movie", 2020)
        uuid_fallback2 = generate_movie_uuid(None, None, "A Movie", 2020)
        self.assertEqual(uuid_fallback1, uuid_fallback2)

    @patch("app.vector_db.qdrant.QdrantClient")
    def test_create_collection(self, mock_client_class):
        """Verify collection creation with specified dimension and distance metric."""
        mock_client_class.return_value = self.real_memory_client
        wrapper = QdrantWrapper(collection_name="movies_test")
        wrapper.connect()

        self.assertTrue(wrapper.create_collection(vector_size=128, distance_metric="Cosine"))
        self.assertTrue(wrapper.collection_exists())

        # Verify info
        col_info = self.real_memory_client.get_collection("movies_test")
        self.assertEqual(col_info.config.params.vectors.size, 128)

    @patch("app.vector_db.qdrant.QdrantClient")
    def test_upload_batch_and_count(self, mock_client_class):
        """Verify uploading batch of points and checking counted points size."""
        mock_client_class.return_value = self.real_memory_client
        wrapper = QdrantWrapper(collection_name="movies_test")
        wrapper.connect()
        wrapper.create_collection(vector_size=4, distance_metric="Cosine")

        points = [
            PointStruct(id=str(uuid.uuid4()), vector=[1.0, 0.0, 0.0, 0.0], payload={"title": f"M{i}"})
            for i in range(10)
        ]
        self.assertTrue(wrapper.upload_batch(points))
        self.assertEqual(wrapper.count_points(), 10)

    def test_pipeline_ingestion_end_to_end(self):
        """Verify ingest_movies_to_qdrant runs end-to-end and uploads points."""
        with self._patch_qdrant_client(), self._patch_settings(), \
             patch("app.vector_db.qdrant.settings", self.mock_settings):
            summary = ingest_movies_to_qdrant(fresh=True, resume=True)

        self.assertEqual(summary["status"], "success")
        self.assertEqual(summary["newly_ingested"], 20)
        self.assertEqual(summary["total_indexed"], 20)

        # Check metadata report saved
        report_path = Path(self.reports_dir) / "qdrant_ingest_report.json"
        self.assertTrue(report_path.exists())
        with open(report_path) as f:
            report = json.load(f)
        self.assertEqual(report["total_indexed"], 20)

    def test_checkpointing_resume(self):
        """Verify checkpoint file skips previously ingested UUIDs."""
        # 1. Run first time with limit 10
        with self._patch_qdrant_client(), self._patch_settings(), \
             patch("app.vector_db.qdrant.settings", self.mock_settings):
            summary1 = ingest_movies_to_qdrant(limit=10, fresh=True, resume=True)

        self.assertEqual(summary1["newly_ingested"], 10)

        # Checkpoint file exists in test directory
        checkpoint_path = Path(self.embeddings_dir) / "checkpoints" / "qdrant_ingest_checkpoint.json"
        self.assertTrue(checkpoint_path.exists())
        with open(checkpoint_path) as f:
            chk = json.load(f)
        self.assertEqual(len(chk), 10)

        # 2. Run second time without fresh, with limit 20
        with self._patch_qdrant_client(), self._patch_settings(), \
             patch("app.vector_db.qdrant.settings", self.mock_settings):
            summary2 = ingest_movies_to_qdrant(limit=20, fresh=False, resume=True)

        # Should only ingest the remaining 10
        self.assertEqual(summary2["newly_ingested"], 10)
        self.assertEqual(summary2["total_indexed"], 20)

    @patch("app.vector_db.qdrant.time.sleep", return_value=None)
    def test_upload_retry_logic(self, mock_sleep):
        """Verify upload_batch retries with exponential backoff on exceptions."""
        mock_client = MagicMock()
        mock_client.upsert.side_effect = Exception("Transient Qdrant Error")

        wrapper = QdrantWrapper(collection_name="movies_test")
        wrapper.client = mock_client

        points = [PointStruct(id=str(uuid.uuid4()), vector=[1.0, 2.0], payload={})]
        
        # Should return False after all 5 retries fail
        success = wrapper.upload_batch(points, max_retries=5)
        self.assertFalse(success)
        self.assertEqual(mock_client.upsert.call_count, 5)


if __name__ == "__main__":
    unittest.main()
