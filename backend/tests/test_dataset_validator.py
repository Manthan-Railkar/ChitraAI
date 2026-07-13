import sys
import unittest
from pathlib import Path
import polars as pl

# Add backend to path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.datasets.loader import DatasetIngestionSystem
from app.datasets.profiler import DatasetProfiler
from app.datasets.validator import DatasetValidator

class TestDatasetValidatorAndProfiler(unittest.TestCase):
    def setUp(self):
        # We load a small mock dataframe to test the profiler directly
        self.mock_df = pl.DataFrame({
            "id": [1, 2, 3, 3, None],
            "title": ["Inception", "Interstellar", "The Dark Knight", "The Dark Knight", "Empty Movie"],
            "rating": [8.8, 8.6, 9.0, 9.0, None],
            "release_date": ["2010-07-16", "2014-11-07", "2008-07-18", "2008-07-18", "invalid-date"]
        })

    def test_mock_profiler(self):
        """Tests the DatasetProfiler functionality with mock data."""
        profile = DatasetProfiler.profile_dataframe("MockMovieTable", self.mock_df)
        
        self.assertEqual(profile["row_count"], 5)
        self.assertEqual(profile["column_count"], 4)
        self.assertEqual(profile["duplicate_statistics"]["duplicate_count"], 1)
        self.assertEqual(profile["missing_values"]["rating"]["count"], 1)
        self.assertEqual(profile["missing_values"]["rating"]["percentage"], 20.0)
        self.assertEqual(len(profile["sample_records"]), 5)
        
        # Verify schema mapping
        self.assertIn("id", profile["schema"])
        self.assertIn("rating", profile["schema"])

    def test_validator_with_mock_dataset(self):
        """Tests specific validator checks using the validator on loaded datasets."""
        # Setup loader to get actual dataset frames (lazy format)
        loader = DatasetIngestionSystem()
        loader.discover_datasets()
        
        # Load only wikipedia dataset (which is fast and has only 1 file)
        wiki_frames = loader.load_dataset("wikipedia", lazy=True)
        
        # Initialize validator with just wikipedia
        validator = DatasetValidator({"wikipedia": wiki_frames})
        wiki_report = validator.validate_wikipedia()
        
        self.assertEqual(wiki_report["dataset"], "Wikipedia")
        self.assertIn("validation_status", wiki_report)
        self.assertIn("issues", wiki_report)
        self.assertIn("statistics", wiki_report)
        self.assertGreater(wiki_report["statistics"]["wiki_rows"], 0)

if __name__ == "__main__":
    unittest.main()
