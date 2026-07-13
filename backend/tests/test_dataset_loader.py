import sys
import unittest
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.datasets.loader import DatasetIngestionSystem, BaseDatasetLoader

class TestDatasetLoader(unittest.TestCase):
    def setUp(self):
        self.ingestion_system = DatasetIngestionSystem()

    def test_discovery(self):
        """Test that dataset folders are discovered correctly."""
        discovered = self.ingestion_system.discover_datasets()
        self.assertGreater(len(discovered), 0, "No dataset folders were discovered.")
        self.assertIn("imdb", self.ingestion_system.loaders)
        self.assertIn("tmdb", self.ingestion_system.loaders)
        self.assertIn("movielens", self.ingestion_system.loaders)
        self.assertIn("wikipedia", self.ingestion_system.loaders)

    def test_validation(self):
        """Test that validation checks pass for the raw directories."""
        self.ingestion_system.discover_datasets()
        validation_results = self.ingestion_system.validate_all()
        
        for name, valid in validation_results.items():
            self.assertTrue(valid, f"Dataset '{name}' failed validation (missing required files).")

    def test_lazy_loading_and_statistics(self):
        """Test that datasets can be loaded lazily and statistics can be generated."""
        self.ingestion_system.discover_datasets()
        
        # We test lazy loading for each discovered dataset
        for name in self.ingestion_system.loaders.keys():
            print(f"Testing loader load for dataset: {name}")
            frames = self.ingestion_system.load_dataset(name, lazy=True)
            self.assertGreater(len(frames), 0, f"No frames loaded for dataset {name}")
            
            # Verify they are LazyFrames (or DataFrames for compressed fallbacks)
            for file_name, frame in frames.items():
                self.assertIsNotNone(frame)
                
            # Get and check stats
            stats = self.ingestion_system.loaders[name].get_summary_statistics()
            self.assertGreater(len(stats), 0)
            for file_name, file_stats in stats.items():
                self.assertIn("row_count", file_stats)
                self.assertIn("column_count", file_stats)
                self.assertIn("estimated_memory_mb", file_stats)
                self.assertIn("disk_size_mb", file_stats)
                self.assertGreaterEqual(file_stats["row_count"], 0)

if __name__ == "__main__":
    unittest.main()
