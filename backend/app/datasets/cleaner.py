import os
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Union, Optional

from loguru import logger
import polars as pl
from app.core.config import settings

# Modular cleaning helpers
def trim_and_clean_text(col_name: str) -> pl.Expr:
    """Trims whitespace and standardizes double-spaces."""
    return pl.col(col_name).str.strip_chars().str.replace_all(r"\s+", " ")

def parse_numeric(col_name: str, dtype: pl.DataType) -> pl.Expr:
    """Attempts casting to numeric, converting unparseable string values to null."""
    return pl.col(col_name).cast(dtype, strict=False)

def parse_iso_date(col_name: str) -> pl.Expr:
    """Parses date string into a Date type, falling back to null if malformed."""
    return pl.col(col_name).str.to_date(format="%Y-%m-%d", strict=False)

def scale_rating(col_name: str, source_max: float, target_max: float = 10.0) -> pl.Expr:
    """Scales ratings to a unified scale (default [0.0, 10.0])."""
    factor = target_max / source_max
    return pl.col(col_name).cast(pl.Float64, strict=False) * factor

def parse_json_list(col_name: str, key_name: str = "name") -> pl.Expr:
    """
    Parses a stringified JSON array natively inside Polars using regex.
    Extracts the values of 'key_name' into a list of strings.
    """
    pattern = rf'"{key_name}":\s*"([^"]*?)"'
    return (
        pl.col(col_name)
        .str.extract_all(pattern)
        .list.eval(pl.element().str.extract(pattern, 1))
    )

def parse_crew_directors(col_name: str) -> pl.Expr:
    """
    Parses the stringified TMDb crew JSON array natively in Polars using regex.
    Filters and extracts name values only for members where job is Director.
    """
    director_object_pattern = r'\{[^{}]*?"job":\s*"Director"[^{}]*?\}'
    name_pattern = r'"name":\s*"([^"]*?)"'
    return (
        pl.col(col_name)
        .str.extract_all(director_object_pattern)
        .list.eval(pl.element().str.extract(name_pattern, 1))
    )


class IMDbCleaner:
    """Cleaner for IMDb dataset."""
    @staticmethod
    def clean_title_basics(lf: pl.LazyFrame) -> pl.LazyFrame:
        # Standardize missing values and clean types
        return lf.with_columns([
            trim_and_clean_text("tconst"),
            trim_and_clean_text("titleType"),
            trim_and_clean_text("primaryTitle"),
            trim_and_clean_text("originalTitle"),
            parse_numeric("isAdult", pl.Int8),
            parse_numeric("startYear", pl.Int32),
            parse_numeric("endYear", pl.Int32),
            parse_numeric("runtimeMinutes", pl.Int32),
            # split genres by comma
            pl.col("genres").str.split(",")
        ]).unique(subset=["tconst"])

    @staticmethod
    def clean_title_ratings(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns([
            trim_and_clean_text("tconst"),
            pl.col("averageRating").cast(pl.Float32, strict=False),
            parse_numeric("numVotes", pl.Int64)
        ]).unique(subset=["tconst"])

    @staticmethod
    def clean_title_principals(lf: pl.LazyFrame) -> pl.LazyFrame:
        # Extract characters from JSON-like bracket string: "[\"Woody\"]" -> ["Woody"]
        char_pattern = r'"([^"]*?)"'
        return lf.with_columns([
            trim_and_clean_text("tconst"),
            parse_numeric("ordering", pl.Int32),
            trim_and_clean_text("nconst"),
            trim_and_clean_text("category"),
            trim_and_clean_text("job"),
            pl.col("characters").str.extract_all(char_pattern).list.eval(pl.element().str.extract(char_pattern, 1))
        ])

    @staticmethod
    def clean_title_crew(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns([
            trim_and_clean_text("tconst"),
            pl.col("directors").str.split(","),
            pl.col("writers").str.split(",")
        ]).unique(subset=["tconst"])

    @staticmethod
    def clean_name_basics(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns([
            trim_and_clean_text("nconst"),
            trim_and_clean_text("primaryName"),
            parse_numeric("birthYear", pl.Int32),
            parse_numeric("deathYear", pl.Int32),
            pl.col("primaryProfession").str.split(","),
            pl.col("knownForTitles").str.split(",")
        ]).unique(subset=["nconst"])


class TMDbCleaner:
    """Cleaner for TMDb dataset."""
    @staticmethod
    def clean_movies_metadata(lf: pl.LazyFrame) -> pl.LazyFrame:
        # Filter out rows with non-numeric IDs (which are corrupted headers/records in Kaggle CSV)
        lf_clean = lf.filter(pl.col("id").cast(pl.String).str.contains(r"^\d+$"))
        
        return lf_clean.with_columns([
            pl.col("id").cast(pl.Int64),
            trim_and_clean_text("title"),
            trim_and_clean_text("overview"),
            parse_iso_date("release_date"),
            parse_json_list("genres", "name"),
            trim_and_clean_text("original_language"),
            parse_json_list("production_companies", "name"),
            pl.col("vote_average").cast(pl.Float32, strict=False),
            parse_numeric("vote_count", pl.Int64)
        ]).unique(subset=["id"])

    @staticmethod
    def clean_credits(lf: pl.LazyFrame) -> pl.LazyFrame:
        # Filter numeric IDs
        lf_clean = lf.filter(pl.col("id").cast(pl.String).str.contains(r"^\d+$"))
        
        return lf_clean.with_columns([
            pl.col("id").cast(pl.Int64),
            parse_json_list("cast", "name").alias("cast_names"),
            parse_crew_directors("crew").alias("director_names")
        ]).unique(subset=["id"])

    @staticmethod
    def clean_keywords(lf: pl.LazyFrame) -> pl.LazyFrame:
        lf_clean = lf.filter(pl.col("id").cast(pl.String).str.contains(r"^\d+$"))
        return lf_clean.with_columns([
            pl.col("id").cast(pl.Int64),
            parse_json_list("keywords", "name").alias("keywords")
        ]).unique(subset=["id"])

    @staticmethod
    def clean_links(lf: pl.LazyFrame) -> pl.LazyFrame:
        # Pad IMDb IDs to tt + 7 digits
        return lf.filter(pl.col("movieId").is_not_null()).with_columns([
            pl.col("movieId").cast(pl.Int64),
            pl.format("tt{}", pl.col("imdbId").cast(pl.String).str.zfill(7)).alias("imdbId"),
            pl.col("tmdbId").cast(pl.Int64, strict=False)
        ]).unique(subset=["movieId"])

    @staticmethod
    def clean_ratings_small(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns([
            pl.col("userId").cast(pl.Int64),
            pl.col("movieId").cast(pl.Int64),
            scale_rating("rating", 5.0, 10.0), # standardizes rating scale to 0.0 - 10.0
            pl.col("timestamp").cast(pl.Int64)
        ])


class MovieLensCleaner:
    """Cleaner for MovieLens dataset."""
    @staticmethod
    def clean_movies(lf: pl.LazyFrame) -> pl.LazyFrame:
        # MovieLens titles look like: "Toy Story (1995)"
        # Extract title and year using regex
        year_pattern = r"\s*\((\d{4})\)\s*$"
        return lf.with_columns([
            pl.col("movieId").cast(pl.Int64),
            pl.col("title").str.replace(year_pattern, "").str.strip_chars().alias("title"),
            pl.col("title").str.extract(year_pattern, 1).cast(pl.Int32, strict=False).alias("release_year"),
            pl.col("genres").str.split("|")
        ]).unique(subset=["movieId"])

    @staticmethod
    def clean_ratings(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns([
            pl.col("userId").cast(pl.Int64),
            pl.col("movieId").cast(pl.Int64),
            scale_rating("rating", 5.0, 10.0), # Scale 0.5-5.0 to 1.0-10.0
            pl.col("timestamp").cast(pl.Int64)
        ])

    @staticmethod
    def clean_tags(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns([
            pl.col("userId").cast(pl.Int64),
            pl.col("movieId").cast(pl.Int64),
            trim_and_clean_text("tag"),
            pl.col("timestamp").cast(pl.Int64)
        ])

    @staticmethod
    def clean_links(lf: pl.LazyFrame) -> pl.LazyFrame:
        # Pad IMDb IDs to tt + 7 digits
        return lf.filter(pl.col("movieId").is_not_null()).with_columns([
            pl.col("movieId").cast(pl.Int64),
            pl.format("tt{}", pl.col("imdbId").cast(pl.String).str.zfill(7)).alias("imdbId"),
            pl.col("tmdbId").cast(pl.Int64, strict=False)
        ]).unique(subset=["movieId"])


class WikipediaCleaner:
    """Cleaner for Wikipedia movie plots dataset."""
    @staticmethod
    def clean_movie_plots(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns([
            parse_numeric("Release Year", pl.Int32),
            trim_and_clean_text("Title"),
            trim_and_clean_text("Origin/Ethnicity"),
            trim_and_clean_text("Director"),
            trim_and_clean_text("Cast"),
            trim_and_clean_text("Genre"),
            trim_and_clean_text("Wiki Page"),
            trim_and_clean_text("Plot")
        ]).unique(subset=["Wiki Page"]) # Dedupes by Wiki page link


class DatasetCleaningPipeline:
    """
    Pipeline coordinator that ingests raw datasets, normalizes them using Polars-optimized
    cleaners, saves output tables in Parquet format, and aggregates corrections statistics.
    """
    def __init__(self, raw_frames: Dict[str, Dict[str, pl.LazyFrame]], output_dir: Optional[str] = None) -> None:
        self.raw_frames = raw_frames
        self.output_dir = Path(output_dir or settings.PROCESSED_DATA_DIR)
        self.summary: Dict[str, Any] = {}
        logger.info(f"DatasetCleaningPipeline initialized. Output directory: {self.output_dir}")

    def run_cleaning(self) -> Dict[str, Any]:
        """Runs the cleaning and normalization steps for every registered dataset."""
        logger.info("Executing dataset cleaning and normalization pipeline...")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean IMDb
        self._clean_imdb_dataset()
        # Clean MovieLens
        self._clean_movielens_dataset()
        # Clean TMDb
        self._clean_tmdb_dataset()
        # Clean Wikipedia
        self._clean_wikipedia_dataset()

        logger.info("Dataset cleaning and normalization pipeline complete.")
        return self.summary

    def _execute_clean_and_save(
        self,
        dataset_name: str,
        file_name: str,
        clean_func: Any,
        output_parquet_name: str
    ) -> None:
        """
        Runs the cleaning function on the source LazyFrame, saves to Parquet,
        and computes cleaning metrics.
        """
        logger.info(f"[{dataset_name}] Cleaning file: '{file_name}'...")
        lf = self.raw_frames.get(dataset_name.lower(), {}).get(file_name)
        if lf is None:
            logger.warning(f"[{dataset_name}] Source file '{file_name}' not found. Skipping.")
            return

        # 1. Compute initial counts and nulls
        initial_rows = lf.select(pl.len()).collect().item()
        initial_nulls = lf.select([pl.col(c).null_count() for c in lf.collect_schema().names()]).collect().row(0)
        initial_null_sum = sum(initial_nulls)


        # 2. Run cleaner function
        cleaned_lf = clean_func(lf)

        # 3. Save as Parquet (collect execution)
        output_path = self.output_dir / output_parquet_name
        logger.debug(f"Saving cleaned dataset to {output_path}...")
        cleaned_df = cleaned_lf.collect()
        cleaned_df.write_parquet(output_path)

        # 4. Compute final statistics
        final_rows = cleaned_df.height
        final_null_sum = sum(cleaned_df.null_count().row(0))
        
        duplicates_removed = max(0, initial_rows - final_rows)
        # Cast conversions and null corrections
        null_corrections = max(0, final_null_sum - initial_null_sum)

        if dataset_name not in self.summary:
            self.summary[dataset_name] = {}

        self.summary[dataset_name][file_name] = {
            "initial_rows": initial_rows,
            "final_rows": final_rows,
            "duplicates_removed": duplicates_removed,
            "type_and_null_corrections": null_corrections,
            "saved_parquet": output_parquet_name
        }

        logger.info(
            f"[{dataset_name}] Cleaned '{file_name}' -> '{output_parquet_name}': "
            f"initial_rows={initial_rows:,}, final_rows={final_rows:,} "
            f"(Removed {duplicates_removed:,} duplicates, Corrected {null_corrections:,} values)"
        )

    def _clean_imdb_dataset(self) -> None:
        self._execute_clean_and_save(
            "IMDb", "title.basics.tsv.gz", IMDbCleaner.clean_title_basics, "imdb_title_basics.parquet"
        )
        self._execute_clean_and_save(
            "IMDb", "title.ratings.tsv.gz", IMDbCleaner.clean_title_ratings, "imdb_title_ratings.parquet"
        )
        self._execute_clean_and_save(
            "IMDb", "title.principals.tsv.gz", IMDbCleaner.clean_title_principals, "imdb_title_principals.parquet"
        )
        self._execute_clean_and_save(
            "IMDb", "title.crew.tsv.gz", IMDbCleaner.clean_title_crew, "imdb_title_crew.parquet"
        )
        self._execute_clean_and_save(
            "IMDb", "name.basics.tsv.gz", IMDbCleaner.clean_name_basics, "imdb_name_basics.parquet"
        )

    def _clean_movielens_dataset(self) -> None:
        self._execute_clean_and_save(
            "MovieLens", "movies.csv", MovieLensCleaner.clean_movies, "movielens_movies.parquet"
        )
        self._execute_clean_and_save(
            "MovieLens", "ratings.csv", MovieLensCleaner.clean_ratings, "movielens_ratings.parquet"
        )
        self._execute_clean_and_save(
            "MovieLens", "tags.csv", MovieLensCleaner.clean_tags, "movielens_tags.parquet"
        )
        self._execute_clean_and_save(
            "MovieLens", "links.csv", MovieLensCleaner.clean_links, "movielens_links.parquet"
        )

    def _clean_tmdb_dataset(self) -> None:
        self._execute_clean_and_save(
            "TMDb", "movies_metadata.csv", TMDbCleaner.clean_movies_metadata, "tmdb_movies_metadata.parquet"
        )
        self._execute_clean_and_save(
            "TMDb", "credits.csv", TMDbCleaner.clean_credits, "tmdb_credits.parquet"
        )
        self._execute_clean_and_save(
            "TMDb", "keywords.csv", TMDbCleaner.clean_keywords, "tmdb_keywords.parquet"
        )
        self._execute_clean_and_save(
            "TMDb", "links.csv", TMDbCleaner.clean_links, "tmdb_links.parquet"
        )
        self._execute_clean_and_save(
            "TMDb", "links_small.csv", TMDbCleaner.clean_links, "tmdb_links_small.parquet"
        )
        self._execute_clean_and_save(
            "TMDb", "ratings_small.csv", TMDbCleaner.clean_ratings_small, "tmdb_ratings_small.parquet"
        )

    def _clean_wikipedia_dataset(self) -> None:
        self._execute_clean_and_save(
            "Wikipedia", "wiki_movie_plots_deduped.csv", WikipediaCleaner.clean_movie_plots, "wikipedia_movie_plots.parquet"
        )
