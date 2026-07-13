import sys
import unittest
from pathlib import Path
import polars as pl

# Add backend to path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.datasets.merger import DatasetMerger
from app.datasets.adapters import CANONICAL_COLUMNS, CANONICAL_SCHEMA

class TestDatasetMerger(unittest.TestCase):
    def setUp(self):
        # Create a temp directory for mock canonical Parquet files
        self.temp_dir = Path(__file__).resolve().parent / "canonical"
        self.temp_dir.mkdir(exist_ok=True)


        # Mock IMDb canonical
        self.imdb_df = pl.DataFrame({
            "imdb_id": ["tt0000001", "tt0000002", "tt0000003"],
            "tmdb_id": [None, None, None],
            "movielens_id": [None, None, None],
            "wiki_page": [None, None, None],
            "title": ["Toy Story", "Jumanji", "IMDb Only Movie"],
            "original_title": ["Toy Story", "Jumanji", "IMDb Only Movie"],
            "overview": [None, None, None],
            "plot_summary": [None, None, None],
            "genres": [["Animation", "Comedy"], ["Adventure"], ["Drama"]],
            "cast": [["Tom Hanks"], ["Robin Williams"], ["Actor C"]],
            "directors": [["John Lasseter"], ["Joe Johnston"], [None]],
            "writers": [[None], [None], [None]],
            "runtime_minutes": [81, 104, 90],
            "release_year": [1995, 1995, 2026],
            "rating_value": [8.3, 7.0, 5.0],
            "vote_count": [10000, 5000, 100], # tt0000003 has 100 votes (>=50 check)
            "popularity": [None, None, None],
            "production_companies": [[None], [None], [None]],
            "languages": [[None], [None], [None]],
            "keywords": [[None], [None], [None]],
            "source_dataset": ["imdb", "imdb", "imdb"]
        })

        # Mock TMDb canonical
        self.tmdb_df = pl.DataFrame({
            "imdb_id": ["tt0000001", None], # tt0000001 linked
            "tmdb_id": [862, 888], # 888 is TMDb-only
            "movielens_id": [None, None],
            "wiki_page": [None, None],
            "title": ["Toy Story", "TMDb Only Movie"],
            "original_title": ["Toy Story", "TMDb Only Movie"],
            "overview": ["A toy story overview.", "TMDb only synopsis."],
            "plot_summary": [None, None],
            "genres": [["Family", "Animation"], ["Action"]],
            "cast": [["Tom Hanks", "Tim Allen"], ["Actor X"]],
            "directors": [["John Lasseter"], ["Director Y"]],
            "writers": [[None], [None]],
            "runtime_minutes": [81, 110],
            "release_year": [1995, 2026],
            "rating_value": [7.9, 6.5],
            "vote_count": [8000, 200],
            "popularity": [25.5, 5.0],
            "production_companies": [["Pixar"], ["Studio Z"]],
            "languages": [["en"], ["fr"]],
            "keywords": [["toys", "boy"], ["action"]],
            "source_dataset": ["tmdb", "tmdb"]
        })

        # Mock MovieLens canonical
        self.movielens_df = pl.DataFrame({
            "imdb_id": ["tt0000001", "tt0000002"],
            "tmdb_id": [862, None],
            "movielens_id": [1, 2],
            "wiki_page": [None, None],
            "title": ["Toy Story", "Jumanji"],
            "original_title": [None, None],
            "overview": [None, None],
            "plot_summary": [None, None],
            "genres": [["Adventure", "Animation"], ["Children"]],
            "cast": [[None], [None]],
            "directors": [[None], [None]],
            "writers": [[None], [None]],
            "runtime_minutes": [None, None],
            "release_year": [1995, 1995],
            "rating_value": [9.0, 7.5],
            "vote_count": [15000, 7000],
            "popularity": [None, None],
            "production_companies": [[None], [None]],
            "languages": [[None], [None]],
            "keywords": [[None], [None]],
            "source_dataset": ["movielens", "movielens"]
        })

        # Mock Wikipedia canonical
        self.wikipedia_df = pl.DataFrame({
            "imdb_id": [None, None],
            "tmdb_id": [None, None],
            "movielens_id": [None, None],
            "wiki_page": ["http://wiki/Toy_Story", "http://wiki/Unlinked"],
            "title": ["Toy Story", "Unlinked Wiki Movie"],
            "original_title": [None, None],
            "overview": [None, None],
            "plot_summary": ["Toy story plot summary.", "Unlinked wiki plot."],
            "genres": [["Animation"], ["Sci-Fi"]],
            "cast": [["Tom Hanks"], ["Cast Z"]],
            "directors": [["John Lasseter"], ["Director W"]],
            "writers": [[None], [None]],
            "runtime_minutes": [None, None],
            "release_year": [1995, 2026],
            "rating_value": [None, None],
            "vote_count": [None, None],
            "popularity": [None, None],
            "production_companies": [[None], [None]],
            "languages": [[None], [None]],
            "keywords": [[None], [None]],
            "source_dataset": ["wikipedia", "wikipedia"]
        })

        # Cast mock dataframes by adding missing columns first to match CANONICAL_SCHEMA
        def prepare_mock_df(df):
            for col, dtype in CANONICAL_SCHEMA.items():
                if col not in df.columns:
                    df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
            df = df.cast(CANONICAL_SCHEMA)
            return df.select(CANONICAL_COLUMNS)

        self.imdb_df = prepare_mock_df(self.imdb_df)
        self.tmdb_df = prepare_mock_df(self.tmdb_df)
        self.movielens_df = prepare_mock_df(self.movielens_df)
        self.wikipedia_df = prepare_mock_df(self.wikipedia_df)



        # Save mock dataframes to temporary Parquet files
        self.imdb_df.write_parquet(self.temp_dir / "imdb_canonical.parquet")

        self.tmdb_df.write_parquet(self.temp_dir / "tmdb_canonical.parquet")
        self.movielens_df.write_parquet(self.temp_dir / "movielens_canonical.parquet")
        self.wikipedia_df.write_parquet(self.temp_dir / "wikipedia_canonical.parquet")

    def tearDown(self):
        # Clean up temporary Parquet files and directory
        for file in self.temp_dir.glob("*.parquet"):
            file.unlink()
        self.temp_dir.rmdir()

    def test_merger_logic(self):
        """Verifies merging groups correctly, calculates weighted average ratings, and maps lists."""
        merger = DatasetMerger(canonical_dir=self.temp_dir.parent, output_dir=self.temp_dir.parent)
        stats = merger.run_merging()

        # Load merged Parquet file
        merged_path = self.temp_dir.parent / "canonical" / "movies_knowledge_base.parquet"
        self.assertTrue(merged_path.exists())
        
        merged = pl.read_parquet(merged_path)
        
        # Verify total unique movies merged:
        # 1. Toy Story (matched across all 4)
        # 2. Jumanji (matched across IMDb, MovieLens)
        # 3. IMDb Only Movie (retained because vote_count=100 >= 50)
        # 4. TMDb Only Movie (retained as TMDb unlinked)
        # 5. Unlinked Wiki Movie (retained as Wikipedia unlinked)
        self.assertEqual(merged.height, 5)

        # Let's verify Toy Story (imdb_id="tt0000001") consolidation
        ts = merged.filter(pl.col("imdb_id") == "tt0000001")
        self.assertEqual(ts.height, 1)

        # 1. Check ID mappings
        self.assertEqual(ts.select("tmdb_id").to_series().to_list()[0], 862)
        self.assertEqual(ts.select("movielens_id").to_series().to_list()[0], 1)
        self.assertEqual(ts.select("wiki_page").to_series().to_list()[0], "http://wiki/Toy_Story")

        # 2. Check source prioritization for strings (TMDb overview took priority)
        self.assertEqual(ts.select("overview").to_series().to_list()[0], "A toy story overview.")
        # Wikipedia plot summary preserved
        self.assertEqual(ts.select("plot_summary").to_series().to_list()[0], "Toy story plot summary.")

        # 3. Check list unions without duplicates
        # IMDb genres: ["Animation", "Comedy"]
        # TMDb genres: ["Family", "Animation"]
        # MovieLens genres: ["Adventure", "Animation"]
        # Wikipedia genres: ["Animation"]
        # Unique Union: ["Action", "Adventure", "Animation", "Comedy", "Family", "Sci-Fi"]?
        # Let's sort and match
        genres = sorted(ts.select("genres").to_series().to_list()[0])
        self.assertEqual(genres, ["Adventure", "Animation", "Comedy", "Family"])

        cast = sorted(ts.select("cast").to_series().to_list()[0])
        self.assertEqual(cast, ["Tim Allen", "Tom Hanks"])

        # 4. Check rating aggregation logic (weighted rating of TMDb + MovieLens + IMDb)
        # TMDb: rating=7.9, votes=8000
        # MovieLens: rating=9.0, votes=15000
        # IMDb: rating=8.3, votes=10000
        # Expected rating_value = (7.9*8000 + 9.0*15000 + 8.3*10000) / (8000 + 15000 + 10000) = 8.5212
        # Expected vote_count = 8000 + 15000 + 10000 = 33000
        # Wait, TMDb is source dataset 1, IMDb is 2, MovieLens is 3. The formula aggregates all rows.
        # Let's check:
        expected_rating = (7.9*8000 + 9.0*15000 + 8.3*10000) / 33000
        actual_rating = ts.select("rating_value").to_series().to_list()[0]
        actual_votes = ts.select("vote_count").to_series().to_list()[0]
        
        self.assertEqual(actual_votes, 33000)
        self.assertAlmostEqual(actual_rating, expected_rating, places=4)




if __name__ == "__main__":
    unittest.main()
