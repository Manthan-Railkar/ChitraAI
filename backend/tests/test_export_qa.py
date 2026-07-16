import sys
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import polars as pl

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.datasets.adapters import CANONICAL_SCHEMA, CANONICAL_COLUMNS
from app.pipelines.export_qa import run_qa_and_export


def _make_test_df(overrides: dict | None = None, n_rows: int = 5) -> pl.DataFrame:
    """Generates a valid test DataFrame conforming to the canonical schema."""
    base = {
        "imdb_id": [f"tt{1000000 + i}" for i in range(n_rows)],
        "tmdb_id": list(range(1, n_rows + 1)),
        "movielens_id": list(range(1, n_rows + 1)),
        "wiki_page": [f"https://en.wikipedia.org/wiki/Movie_{i}" for i in range(n_rows)],
        "title": [f"Movie {i}" for i in range(n_rows)],
        "original_title": [f"Movie {i}" for i in range(n_rows)],
        "overview": [f"Overview for movie {i}" for i in range(n_rows)],
        "plot_summary": [f"Detailed plot summary for movie {i}." for i in range(n_rows)],
        "genres": [["Drama", "Thriller"] for _ in range(n_rows)],
        "cast": [["Actor A", "Actor B"] for _ in range(n_rows)],
        "directors": [["Director A"] for _ in range(n_rows)],
        "writers": [["Writer A"] for _ in range(n_rows)],
        "runtime_minutes": [120 + i for i in range(n_rows)],
        "release_year": [2000 + i for i in range(n_rows)],
        "rating_value": [7.0 + i * 0.3 for i in range(n_rows)],
        "vote_count": [1000 * (i + 1) for i in range(n_rows)],
        "popularity": [50.0 + i for i in range(n_rows)],
        "production_companies": [["Studio A"] for _ in range(n_rows)],
        "languages": [["en"] for _ in range(n_rows)],
        "keywords": [["thriller", "mystery"] for _ in range(n_rows)],
        "source_dataset": ["imdb"] * n_rows,
        "poster_path": ["/poster.jpg"] * n_rows,
        "backdrop_path": ["/backdrop.jpg"] * n_rows,
        "trailer_url": ["https://youtube.com/watch?v=abc"] * n_rows,
        "streaming_providers": [["Netflix"]] * n_rows,
        "collection_name": [None] * n_rows,
        "certification": ["PG-13"] * n_rows,
        "tagline": ["Tagline text"] * n_rows,
        "document": [f"Title: Movie {i} (200{i})\nGenres: Drama, Thriller" for i in range(n_rows)],
    }
    if overrides:
        base.update(overrides)
    df = pl.DataFrame(base).cast(CANONICAL_SCHEMA)
    return df


class TestExportQA(unittest.TestCase):
    """Tests for the Quality Assurance and Export Pipeline."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.processed_dir = Path(self.tmpdir) / "processed" / "canonical"
        self.processed_dir.mkdir(parents=True)
        self.merged_dir = Path(self.tmpdir) / "merged"
        self.merged_dir.mkdir(parents=True)
        self.reports_dir = Path(self.tmpdir) / "reports"
        self.reports_dir.mkdir(parents=True)

    def _write_and_run(self, df: pl.DataFrame) -> dict:
        """Writes a test DataFrame to the canonical KB parquet and runs QA."""
        kb_path = self.processed_dir / "movies_knowledge_base.parquet"
        df.write_parquet(kb_path)

        with patch("app.pipelines.export_qa.settings") as mock_settings:
            mock_settings.PROCESSED_DATA_DIR = str(Path(self.tmpdir) / "processed")
            mock_settings.MERGED_DATA_DIR = str(self.merged_dir)
            mock_settings.REPORTS_DIR = str(self.reports_dir)
            return run_qa_and_export()

    def test_clean_data_passes_all_checks(self):
        """A fully valid dataset should pass with zero quality issues."""
        df = _make_test_df()
        report = self._write_and_run(df)

        stats = report["dataset_statistics"]
        self.assertEqual(stats["initial_rows"], 5)
        self.assertEqual(stats["final_rows"], 5)
        self.assertEqual(stats["removed_rows"], 0)
        self.assertTrue(stats["schema_consistent"])

        metrics = report["quality_metrics"]
        self.assertEqual(metrics["duplicate_imdb_ids"], 0)
        self.assertEqual(metrics["missing_titles"], 0)
        self.assertEqual(metrics["invalid_ratings"], 0)
        self.assertEqual(metrics["orphan_records"], 0)

    def test_detects_duplicate_imdb_ids(self):
        """Duplicate IMDb IDs should be counted in quality metrics."""
        df = _make_test_df(overrides={
            "imdb_id": ["tt1000000", "tt1000000", "tt1000002", "tt1000003", "tt1000004"],
        })
        report = self._write_and_run(df)
        self.assertEqual(report["quality_metrics"]["duplicate_imdb_ids"], 2)

    def test_detects_missing_titles(self):
        """Movies with null or empty titles should be detected and filtered out."""
        df = _make_test_df(overrides={
            "title": ["Good Movie", None, "", "  ", "Another Movie"],
        })
        report = self._write_and_run(df)
        # null, empty, whitespace-only titles are all "missing"
        self.assertGreaterEqual(report["quality_metrics"]["missing_titles"], 2)
        # Removed count should include filtered records
        self.assertGreater(report["dataset_statistics"]["removed_rows"], 0)

    def test_detects_invalid_imdb_format(self):
        """IMDb IDs not matching tt\\d{7,8} should be flagged."""
        df = _make_test_df(overrides={
            "imdb_id": ["tt1234567", "nm9999999", "tt12", "invalid", "tt12345678"],
        })
        report = self._write_and_run(df)
        # nm9999999, tt12, and "invalid" are invalid format
        self.assertEqual(report["quality_metrics"]["invalid_imdb_format"], 3)

    def test_detects_invalid_ratings(self):
        """Ratings outside [1.0, 10.0] should be detected and those records filtered."""
        df = _make_test_df(overrides={
            "rating_value": [7.5, 0.5, 10.5, -1.0, 8.0],
        })
        report = self._write_and_run(df)
        self.assertEqual(report["quality_metrics"]["invalid_ratings"], 3)
        # Records with invalid ratings are filtered out
        self.assertEqual(report["dataset_statistics"]["removed_rows"], 3)

    def test_detects_orphan_records(self):
        """Records with no identifiers should be flagged and filtered."""
        df = _make_test_df(overrides={
            "imdb_id": [None, "tt1000001", "tt1000002", None, "tt1000004"],
            "tmdb_id": [None, 2, 3, None, 5],
            "movielens_id": [None, 2, 3, None, 5],
            "wiki_page": [None, "url1", "url2", None, "url4"],
        })
        report = self._write_and_run(df)
        self.assertEqual(report["quality_metrics"]["orphan_records"], 2)
        self.assertEqual(report["dataset_statistics"]["removed_rows"], 2)

    def test_genre_distribution(self):
        """Genre distribution should correctly count exploded genre occurrences."""
        df = _make_test_df(overrides={
            "genres": [
                ["Drama"],
                ["Drama", "Action"],
                ["Comedy"],
                ["Drama", "Comedy"],
                ["Action"],
            ],
        })
        report = self._write_and_run(df)
        genres = report["distributions"]["genres"]
        self.assertEqual(genres.get("Drama"), 3)
        self.assertEqual(genres.get("Action"), 2)
        self.assertEqual(genres.get("Comedy"), 2)

    def test_rating_distribution_bins(self):
        """Rating distribution should bucket values into floor-based bins."""
        df = _make_test_df(overrides={
            "rating_value": [7.1, 7.9, 8.0, 8.5, 9.0],
        })
        report = self._write_and_run(df)
        bins = report["distributions"]["ratings"]
        self.assertIn("7.0-8.0", bins)
        self.assertEqual(bins["7.0-8.0"], 2)
        self.assertIn("8.0-9.0", bins)
        self.assertEqual(bins["8.0-9.0"], 2)

    def test_missing_value_summary(self):
        """Missing value summary should report correct null counts and percentages."""
        df = _make_test_df(overrides={
            "overview": [None, "overview", None, None, "overview"],
        })
        report = self._write_and_run(df)
        overview_stats = report["missing_value_summary"]["overview"]
        self.assertEqual(overview_stats["null_count"], 3)
        self.assertAlmostEqual(overview_stats["null_percentage"], 60.0, places=1)

    def test_exports_parquet_csv_json(self):
        """Exported files should exist in all three formats after QA."""
        df = _make_test_df()
        self._write_and_run(df)

        self.assertTrue((self.merged_dir / "movies_knowledge_base.parquet").exists())
        self.assertTrue((self.merged_dir / "movies_knowledge_base.csv").exists())
        self.assertTrue((self.merged_dir / "movies_knowledge_base.json").exists())

    def test_exported_parquet_row_count(self):
        """Exported Parquet should contain only clean rows."""
        df = _make_test_df(overrides={
            "title": ["Good", None, "Also Good", "Fine", "OK"],
        })
        report = self._write_and_run(df)
        exported = pl.read_parquet(self.merged_dir / "movies_knowledge_base.parquet")
        self.assertEqual(exported.height, report["dataset_statistics"]["final_rows"])

    def test_qa_report_saved_to_reports_dir(self):
        """The QA report JSON should be written to the reports directory."""
        df = _make_test_df()
        self._write_and_run(df)
        report_file = self.reports_dir / "movies_validation_report.json"
        self.assertTrue(report_file.exists())
        with open(report_file) as f:
            saved = json.load(f)
        self.assertIn("dataset_statistics", saved)
        self.assertIn("quality_metrics", saved)
        self.assertIn("distributions", saved)
        self.assertIn("missing_value_summary", saved)

    def test_schema_consistency_check(self):
        """Schema consistency flag should be True for correctly shaped data."""
        df = _make_test_df()
        report = self._write_and_run(df)
        self.assertTrue(report["dataset_statistics"]["schema_consistent"])

    def test_empty_genres_detected(self):
        """Movies with null or empty genre lists should be flagged."""
        df = _make_test_df(overrides={
            "genres": [["Drama"], None, [], ["Action"], None],
        })
        report = self._write_and_run(df)
        self.assertEqual(report["quality_metrics"]["missing_genres"], 3)


if __name__ == "__main__":
    unittest.main()
