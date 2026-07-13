import sys
import unittest
from pathlib import Path
import polars as pl

# Add backend to path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.datasets.adapters import (
    CANONICAL_COLUMNS,
    CANONICAL_SCHEMA,
    IMDbAdapter,
    TMDbAdapter,
    MovieLensAdapter,
    WikipediaAdapter
)

class TestCanonicalAdapters(unittest.TestCase):
    def test_canonical_schema_conformity(self):
        """Verifies that all adapters output the exact same column set, types, and ordering."""
        # Setup minimal mock dataframes for each source
        imdb_data = {
            "title.basics.tsv.gz": pl.DataFrame({
                "tconst": ["tt0000001"],
                "primaryTitle": ["Test IMDb Movie"],
                "originalTitle": ["Test Original"],
                "genres": [["Drama"]],
                "runtimeMinutes": [120],
                "startYear": [2026]
            }).lazy(),
            "title.ratings.tsv.gz": pl.DataFrame({
                "tconst": ["tt0000001"],
                "averageRating": [8.5],
                "numVotes": [1000]
            }).lazy()
        }

        tmdb_data = {
            "movies_metadata.csv": pl.DataFrame({
                "id": [123],
                "imdb_id": ["tt0000001"],
                "title": ["Test TMDb Movie"],
                "original_title": ["Test Original"],
                "overview": ["Test Overview"],
                "genres": [["Drama"]],
                "runtime": [120],
                "release_date": ["2026-07-13"],
                "vote_average": [8.5],
                "vote_count": [1000],
                "popularity": [1.5],
                "production_companies": [["Universal"]],
                "original_language": ["en"]
            }).with_columns(pl.col("release_date").str.to_date()).lazy()
        }

        movielens_data = {
            "movies.csv": pl.DataFrame({
                "movieId": [1],
                "title": ["Test MovieLens Movie"],
                "release_year": [2026],
                "genres": [["Drama"]]
            }).lazy()
        }

        wikipedia_data = {
            "wiki_movie_plots_deduped.csv": pl.DataFrame({
                "Wiki Page": ["https://en.wikipedia.org/wiki/Test"],
                "Title": ["Test Wikipedia Movie"],
                "Plot": ["Test detailed plot summary."],
                "Release Year": [2026],
                "Genre": ["drama"],
                "Director": ["John Doe"],
                "Cast": ["Jane Smith"]
            }).lazy()
        }

        # Adapt all sources
        imdb_adapted = IMDbAdapter().adapt(imdb_data).collect()
        tmdb_adapted = TMDbAdapter().adapt(tmdb_data).collect()
        movielens_adapted = MovieLensAdapter().adapt(movielens_data).collect()
        wikipedia_adapted = WikipediaAdapter().adapt(wikipedia_data).collect()

        # Assert all schemas match CANONICAL_SCHEMA exactly (column names, types, order)
        for name, df in [
            ("IMDb", imdb_adapted),
            ("TMDb", tmdb_adapted),
            ("MovieLens", movielens_adapted),
            ("Wikipedia", wikipedia_adapted)
        ]:
            self.assertEqual(df.columns, CANONICAL_COLUMNS, f"{name} columns order/names do not match.")
            for col in CANONICAL_COLUMNS:
                expected_type = CANONICAL_SCHEMA[col]
                actual_type = df.schema[col]
                self.assertEqual(actual_type, expected_type, f"{name} column '{col}' has type {actual_type}, expected {expected_type}")

    def test_imdb_adapter_name_resolution(self):
        """Verifies IMDb adapter resolves directors, writers, and cast nconst identifiers to primaryName strings."""
        imdb_data = {
            "title.basics.tsv.gz": pl.DataFrame({
                "tconst": ["tt111"], "primaryTitle": ["Movie"], "originalTitle": ["Movie"],
                "genres": [["Drama"]], "runtimeMinutes": [90], "startYear": [2026]
            }).lazy(),
            "title.crew.tsv.gz": pl.DataFrame({
                "tconst": ["tt111"], "directors": [["nm100"]], "writers": [["nm200"]]
            }).lazy(),
            "title.principals.tsv.gz": pl.DataFrame({
                "tconst": ["tt111"], "ordering": [1], "nconst": ["nm300"], "category": ["actor"]
            }).lazy(),
            "name.basics.tsv.gz": pl.DataFrame([
                {"nconst": "nm100", "primaryName": "Director Name"},
                {"nconst": "nm200", "primaryName": "Writer Name"},
                {"nconst": "nm300", "primaryName": "Actor Name"}
            ]).lazy()
        }

        df = IMDbAdapter().adapt(imdb_data).collect()
        
        self.assertEqual(df.select("directors").to_series().to_list()[0], ["Director Name"])
        self.assertEqual(df.select("writers").to_series().to_list()[0], ["Writer Name"])
        self.assertEqual(df.select("cast").to_series().to_list()[0], ["Actor Name"])

    def test_movielens_adapter_rating_aggregations(self):
        """Verifies MovieLens adapter aggregates rating value and vote count statistics per movie."""
        movielens_data = {
            "movies.csv": pl.DataFrame({
                "movieId": [1], "title": ["Toy Story"], "release_year": [1995], "genres": [["Animation"]]
            }).lazy(),
            "ratings.csv": pl.DataFrame([
                {"userId": 1, "movieId": 1, "rating": 8.0, "timestamp": 123},
                {"userId": 2, "movieId": 1, "rating": 10.0, "timestamp": 124}
            ]).lazy(),
            "links.csv": pl.DataFrame({
                "movieId": [1], "imdbId": ["0114709"], "tmdbId": [862]
            }).lazy()
        }

        df = MovieLensAdapter().adapt(movielens_data).collect()
        
        self.assertEqual(df.select("rating_value").to_series().to_list()[0], 9.0) # Average of 8 and 10
        self.assertEqual(df.select("vote_count").to_series().to_list()[0], 2)

    def test_wikipedia_genre_splitting(self):
        """Verifies Wikipedia adapter splits complex slash/comma genre strings and trims list items."""
        wikipedia_data = {
            "wiki_movie_plots_deduped.csv": pl.DataFrame({
                "Wiki Page": ["url"], "Title": ["Movie"], "Plot": ["Plot"], "Release Year": [2026],
                "Genre": ["drama/romance, comedy"], "Director": ["John Doe, Jane Doe"], "Cast": ["Cast A, Cast B"]
            }).lazy()
        }

        df = WikipediaAdapter().adapt(wikipedia_data).collect()
        
        self.assertEqual(df.select("genres").to_series().to_list()[0], ["drama", "romance", "comedy"])
        self.assertEqual(df.select("directors").to_series().to_list()[0], ["John Doe", "Jane Doe"])
        self.assertEqual(df.select("cast").to_series().to_list()[0], ["Cast A", "Cast B"])


if __name__ == "__main__":
    unittest.main()
