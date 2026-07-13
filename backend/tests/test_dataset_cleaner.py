import sys
import unittest
from pathlib import Path
import polars as pl

# Add backend to path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.datasets.cleaner import (
    trim_and_clean_text,
    parse_numeric,
    parse_iso_date,
    scale_rating,
    parse_json_list,
    parse_crew_directors,
    MovieLensCleaner
)

class TestDatasetCleanerHelpers(unittest.TestCase):
    def test_trim_and_clean_text(self):
        df = pl.DataFrame({"text": ["  hello   world  ", "clean text"]})
        res = df.select(trim_and_clean_text("text")).to_series().to_list()
        self.assertEqual(res, ["hello world", "clean text"])

    def test_parse_numeric(self):
        df = pl.DataFrame({"num": ["123", "abc", "45.6", None]})
        res = df.select(parse_numeric("num", pl.Int32)).to_series().to_list()
        self.assertEqual(res, [123, None, None, None])

    def test_parse_iso_date(self):
        df = pl.DataFrame({"date": ["2026-07-13", "invalid-date", None, "2026/07/13"]})
        res = df.select(parse_iso_date("date")).to_series().to_list()
        # The successfully parsed date is represented as a Python datetime.date object
        self.assertIsNotNone(res[0])
        self.assertEqual(res[0].year, 2026)
        self.assertEqual(res[0].month, 7)
        self.assertEqual(res[0].day, 13)
        self.assertIsNone(res[1])
        self.assertIsNone(res[2])
        self.assertIsNone(res[3]) # Malformed format

    def test_scale_rating(self):
        df = pl.DataFrame({"rating": [2.5, 5.0, 0.5, None]})
        res = df.select(scale_rating("rating", 5.0, 10.0)).to_series().to_list()
        self.assertEqual(res, [5.0, 10.0, 1.0, None])

    def test_parse_json_list(self):
        df = pl.DataFrame({
            "genres": ['[{"id": 18, "name": "Drama"}, {"id": 80, "name": "Crime"}]']
        })
        res = df.select(parse_json_list("genres", "name")).to_series().to_list()
        self.assertEqual(res, [["Drama", "Crime"]])

    def test_parse_crew_directors(self):
        df = pl.DataFrame({
            "crew": ['[{"job": "Director", "name": "John Lasseter"}, {"job": "Screenplay", "name": "Joss Whedon"}]']
        })
        res = df.select(parse_crew_directors("crew")).to_series().to_list()
        self.assertEqual(res, [["John Lasseter"]])

    def test_movielens_title_year_regex(self):
        df = pl.DataFrame({
            "movieId": [1, 2],
            "title": ["Toy Story (1995)", "No Year Movie"],
            "genres": ["Animation|Comedy", "Drama"]
        })
        cleaned_lf = MovieLensCleaner.clean_movies(df.lazy())
        cleaned_df = cleaned_lf.collect()
        
        self.assertEqual(cleaned_df.filter(pl.col("movieId") == 1).select("title").to_series().to_list()[0], "Toy Story")
        self.assertEqual(cleaned_df.filter(pl.col("movieId") == 1).select("release_year").to_series().to_list()[0], 1995)
        
        self.assertEqual(cleaned_df.filter(pl.col("movieId") == 2).select("title").to_series().to_list()[0], "No Year Movie")
        self.assertIsNone(cleaned_df.filter(pl.col("movieId") == 2).select("release_year").to_series().to_list()[0])

if __name__ == "__main__":
    unittest.main()
