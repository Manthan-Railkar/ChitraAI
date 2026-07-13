"""
Unit tests for the ChitraAI Embedding Generation Pipeline.

Tests cover embedding service initialization, batch encoding, checkpoint
recovery, pipeline execution, metadata reports, and empty document filtering.
All tests use mocked SentenceTransformer models to avoid downloading real weights.
"""

import sys
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import polars as pl

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.datasets.adapters import CANONICAL_SCHEMA, CANONICAL_COLUMNS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MOCK_DIM = 768

# Patch target: the lazy import inside _load_model uses
# `from sentence_transformers import SentenceTransformer`
# so we patch the module-level class.
PATCH_TARGET = "sentence_transformers.SentenceTransformer"


def _make_mock_sentence_transformer():
    """Creates a mock SentenceTransformer that returns deterministic embeddings."""
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = MOCK_DIM
    mock_model.max_seq_length = 512

    def mock_encode(texts, **kwargs):
        n = len(texts)
        # Generate deterministic normalized vectors
        rng = np.random.RandomState(42)
        vecs = rng.randn(n, MOCK_DIM).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    mock_model.encode = mock_encode
    return mock_model


def _make_test_kb(n_rows: int = 20) -> pl.DataFrame:
    """Creates a minimal test knowledge base DataFrame."""
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


def _patch_settings(tmpdir: str, embeddings_dir: str, reports_dir: str):
    """Returns a patch context manager for settings used by the pipeline."""
    mock_settings = MagicMock()
    mock_settings.PROCESSED_DATA_DIR = str(Path(tmpdir) / "processed")
    mock_settings.EMBEDDINGS_DIR = embeddings_dir
    mock_settings.REPORTS_DIR = reports_dir
    mock_settings.EMBEDDING_MODEL = "mock-model"
    mock_settings.EMBEDDING_BATCH_SIZE = 5
    mock_settings.EMBEDDING_CHECKPOINT_INTERVAL = 2
    mock_settings.DEVICE = "cpu"
    mock_settings.LOG_LEVEL = "INFO"
    return mock_settings


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEmbeddingService(unittest.TestCase):
    """Tests for the EmbeddingService class."""

    @patch(PATCH_TARGET)
    def test_encode_batch_shape_and_normalization(self, mock_st_class):
        """Verify batch encoding returns correct shape and L2-normalized vectors."""
        mock_st_class.return_value = _make_mock_sentence_transformer()

        from app.services.embedding_service import EmbeddingService
        service = EmbeddingService(model_name="mock-model", device="cpu")

        texts = ["Hello world", "Another sentence", "Third document"]
        embeddings = service.encode_batch(texts)

        # Shape check
        self.assertEqual(embeddings.shape, (3, MOCK_DIM))
        self.assertEqual(embeddings.dtype, np.float32)

        # Normalization check (L2 norms should be ≈ 1.0)
        norms = np.linalg.norm(embeddings, axis=1)
        for norm in norms:
            self.assertAlmostEqual(norm, 1.0, places=4)

    @patch(PATCH_TARGET)
    def test_lazy_loading(self, mock_st_class):
        """Verify model is not loaded until first encode call."""
        mock_st_class.return_value = _make_mock_sentence_transformer()

        from app.services.embedding_service import EmbeddingService
        service = EmbeddingService(model_name="mock-model", device="cpu")

        # Model should NOT be loaded yet
        self.assertFalse(service.is_loaded)
        mock_st_class.assert_not_called()

        # Trigger loading via encode
        service.encode_batch(["test"])

        # Model should now be loaded
        self.assertTrue(service.is_loaded)
        mock_st_class.assert_called_once()


class TestBuildEmbeddingsPipeline(unittest.TestCase):
    """Tests for the embedding generation pipeline."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.processed_dir = Path(self.tmpdir) / "processed" / "canonical"
        self.processed_dir.mkdir(parents=True)
        self.embeddings_dir = Path(self.tmpdir) / "embeddings"
        self.embeddings_dir.mkdir(parents=True)
        self.reports_dir = Path(self.tmpdir) / "reports"
        self.reports_dir.mkdir(parents=True)

        # Write test knowledge base
        self.test_df = _make_test_kb(n_rows=20)
        self.test_df.write_parquet(self.processed_dir / "movies_knowledge_base.parquet")

        # Build mock settings
        self.mock_settings = _patch_settings(
            self.tmpdir, str(self.embeddings_dir), str(self.reports_dir)
        )

    @patch(PATCH_TARGET)
    def test_pipeline_generates_embeddings(self, mock_st_class):
        """End-to-end: verify output Parquet schema, row count, and embedding shape."""
        mock_st_class.return_value = _make_mock_sentence_transformer()

        with patch("app.pipelines.build_embeddings.settings", self.mock_settings), \
             patch("app.services.embedding_service.settings", self.mock_settings):
            from app.pipelines.build_embeddings import build_movie_embeddings
            summary = build_movie_embeddings(fresh=True, batch_size=5, device="cpu")

        # Check output file exists
        output_path = self.embeddings_dir / "movie_embeddings.parquet"
        self.assertTrue(output_path.exists())

        # Check row count
        result = pl.read_parquet(output_path)
        self.assertEqual(result.height, 20)

        # Check schema
        self.assertIn("embedding", result.columns)
        self.assertIn("imdb_id", result.columns)
        self.assertIn("title", result.columns)

        # Check embedding dimension
        sample_emb = result.select("embedding").row(0)[0]
        self.assertEqual(len(sample_emb), MOCK_DIM)

        # Check summary
        self.assertEqual(summary["total_movies_embedded"], 20)
        self.assertEqual(summary["embedding_dimension"], MOCK_DIM)

    @patch(PATCH_TARGET)
    def test_checkpoint_recovery(self, mock_st_class):
        """Verify that pre-existing checkpoints are resumed correctly."""
        mock_st_class.return_value = _make_mock_sentence_transformer()

        # Create a partial checkpoint with the first 10 movies
        checkpoint_dir = self.embeddings_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        partial_df = _make_test_kb(n_rows=10)
        rng = np.random.RandomState(42)
        vecs = rng.randn(10, MOCK_DIM).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms
        embedding_lists = [v.tolist() for v in vecs]

        checkpoint_data = partial_df.select(
            ["imdb_id", "tmdb_id", "movielens_id", "title", "release_year"]
        ).with_columns(
            pl.Series("embedding", embedding_lists, dtype=pl.List(pl.Float32))
        )
        checkpoint_data.write_parquet(checkpoint_dir / "embedding_checkpoint.parquet")

        with patch("app.pipelines.build_embeddings.settings", self.mock_settings), \
             patch("app.services.embedding_service.settings", self.mock_settings):
            from app.pipelines.build_embeddings import build_movie_embeddings
            summary = build_movie_embeddings(resume=True, batch_size=5, device="cpu")

        # Should have embedded only the remaining 10
        self.assertEqual(summary["newly_embedded"], 10)

        # Final output should contain all 20
        result = pl.read_parquet(self.embeddings_dir / "movie_embeddings.parquet")
        self.assertEqual(result.height, 20)

    @patch(PATCH_TARGET)
    def test_metadata_report(self, mock_st_class):
        """Verify the JSON metadata report contains expected fields."""
        mock_st_class.return_value = _make_mock_sentence_transformer()

        with patch("app.pipelines.build_embeddings.settings", self.mock_settings), \
             patch("app.services.embedding_service.settings", self.mock_settings):
            from app.pipelines.build_embeddings import build_movie_embeddings
            build_movie_embeddings(fresh=True, batch_size=10, device="cpu")

        meta_path = self.embeddings_dir / "embedding_metadata.json"
        self.assertTrue(meta_path.exists())

        with open(meta_path) as f:
            meta = json.load(f)

        self.assertIn("model_name", meta)
        self.assertIn("embedding_dimension", meta)
        self.assertIn("total_movies_embedded", meta)
        self.assertIn("device", meta)
        self.assertIn("total_time_seconds", meta)
        self.assertIn("output_file", meta)
        self.assertEqual(meta["embedding_dimension"], MOCK_DIM)
        self.assertEqual(meta["total_movies_embedded"], 20)

    @patch(PATCH_TARGET)
    def test_empty_document_filtering(self, mock_st_class):
        """Verify rows with null/empty documents are excluded from embedding."""
        mock_st_class.return_value = _make_mock_sentence_transformer()

        # Create KB with some null/empty documents
        df = _make_test_kb(n_rows=10)
        df = df.with_columns(
            pl.when(pl.col("imdb_id").is_in(["tt1000000", "tt1000001", "tt1000002"]))
            .then(pl.lit(None))
            .otherwise(pl.col("document"))
            .alias("document")
        )
        df.write_parquet(
            self.processed_dir / "movies_knowledge_base.parquet"
        )

        with patch("app.pipelines.build_embeddings.settings", self.mock_settings), \
             patch("app.services.embedding_service.settings", self.mock_settings):
            from app.pipelines.build_embeddings import build_movie_embeddings
            summary = build_movie_embeddings(fresh=True, batch_size=5, device="cpu")

        # Should only have embedded 7 movies (10 - 3 nulls)
        result = pl.read_parquet(self.embeddings_dir / "movie_embeddings.parquet")
        self.assertEqual(result.height, 7)
        self.assertEqual(summary["total_movies_embedded"], 7)

    @patch(PATCH_TARGET)
    def test_report_saved_to_reports_dir(self, mock_st_class):
        """Verify the embedding report is also saved to the reports directory."""
        mock_st_class.return_value = _make_mock_sentence_transformer()

        with patch("app.pipelines.build_embeddings.settings", self.mock_settings), \
             patch("app.services.embedding_service.settings", self.mock_settings):
            from app.pipelines.build_embeddings import build_movie_embeddings
            build_movie_embeddings(fresh=True, batch_size=10, device="cpu")

        report_path = self.reports_dir / "embedding_generation_report.json"
        self.assertTrue(report_path.exists())

        with open(report_path) as f:
            report = json.load(f)
        self.assertIn("timestamp", report)
        self.assertIn("model_name", report)

    @patch(PATCH_TARGET)
    def test_limit_flag(self, mock_st_class):
        """Verify --limit flag caps the number of movies processed."""
        mock_st_class.return_value = _make_mock_sentence_transformer()

        with patch("app.pipelines.build_embeddings.settings", self.mock_settings), \
             patch("app.services.embedding_service.settings", self.mock_settings):
            from app.pipelines.build_embeddings import build_movie_embeddings
            summary = build_movie_embeddings(limit=5, fresh=True, batch_size=5, device="cpu")

        result = pl.read_parquet(self.embeddings_dir / "movie_embeddings.parquet")
        self.assertEqual(result.height, 5)
        self.assertEqual(summary["total_movies_embedded"], 5)


if __name__ == "__main__":
    unittest.main()
