from abc import ABC, abstractmethod
from typing import Dict
from loguru import logger
import polars as pl

# Canonical schema definition
CANONICAL_COLUMNS = [
    "imdb_id",
    "tmdb_id",
    "movielens_id",
    "wiki_page",
    "title",
    "original_title",
    "overview",
    "plot_summary",
    "genres",
    "cast",
    "directors",
    "writers",
    "runtime_minutes",
    "release_year",
    "rating_value",
    "vote_count",
    "popularity",
    "production_companies",
    "languages",
    "keywords",
    "source_dataset",
    "poster_path",
    "backdrop_path",
    "trailer_url",
    "streaming_providers",
    "collection_name",
    "certification",
    "document"
]

CANONICAL_SCHEMA = {
    "imdb_id": pl.String,
    "tmdb_id": pl.Int64,
    "movielens_id": pl.Int64,
    "wiki_page": pl.String,
    "title": pl.String,
    "original_title": pl.String,
    "overview": pl.String,
    "plot_summary": pl.String,
    "genres": pl.List(pl.String),
    "cast": pl.List(pl.String),
    "directors": pl.List(pl.String),
    "writers": pl.List(pl.String),
    "runtime_minutes": pl.Int32,
    "release_year": pl.Int32,
    "rating_value": pl.Float32,
    "vote_count": pl.Int64,
    "popularity": pl.Float32,
    "production_companies": pl.List(pl.String),
    "languages": pl.List(pl.String),
    "keywords": pl.List(pl.String),
    "source_dataset": pl.String,
    "poster_path": pl.String,
    "backdrop_path": pl.String,
    "trailer_url": pl.String,
    "streaming_providers": pl.List(pl.String),
    "collection_name": pl.String,
    "certification": pl.String,
    "document": pl.String
}




class BaseMovieAdapter(ABC):
    """Abstract base class for dataset-specific canonical schema adapters."""
    @abstractmethod
    def adapt(self, datasets: Dict[str, pl.LazyFrame]) -> pl.LazyFrame:
        """Transforms raw cleaned dataset components into the canonical movie schema."""
        pass

    def enforce_canonical_schema(self, lf: pl.LazyFrame, source_dataset: str) -> pl.LazyFrame:
        """
        Guarantees that the output LazyFrame conforms exactly to the canonical schema
        by adding missing columns, casting types, and reordering columns.
        """
        # Inject the source dataset flag
        lf = lf.with_columns(pl.lit(source_dataset).alias("source_dataset"))
        
        select_exprs = []
        schema = lf.collect_schema()
        
        for col in CANONICAL_COLUMNS:
            target_type = CANONICAL_SCHEMA[col]
            if col in schema:
                select_exprs.append(pl.col(col).cast(target_type).alias(col))
            else:
                select_exprs.append(pl.lit(None).cast(target_type).alias(col))
                
        return lf.select(select_exprs)


class IMDbAdapter(BaseMovieAdapter):
    """Adapter to map IMDb tables (basics, ratings, crew, principals, names) to the canonical schema."""
    def adapt(self, datasets: Dict[str, pl.LazyFrame]) -> pl.LazyFrame:
        logger.info("[IMDbAdapter] Translating tables into canonical schema...")
        basics = datasets.get("title.basics.tsv.gz")
        ratings = datasets.get("title.ratings.tsv.gz")
        crew = datasets.get("title.crew.tsv.gz")
        principals = datasets.get("title.principals.tsv.gz")
        names = datasets.get("name.basics.tsv.gz")

        if basics is None:
            raise ValueError("IMDb title.basics is required for mapping.")

        # 1. Resolve crew names (directors and writers) using names basics
        directors_resolved = None
        writers_resolved = None
        if crew is not None and names is not None:
            directors_resolved = (
                crew.select(["tconst", "directors"])
                .explode("directors")
                .filter(pl.col("directors").is_not_null() & (pl.col("directors") != ""))
                .join(names.select(["nconst", "primaryName"]), left_on="directors", right_on="nconst", how="left")
                .group_by("tconst")
                .agg(pl.col("primaryName").alias("directors"))
            )
            
            writers_resolved = (
                crew.select(["tconst", "writers"])
                .explode("writers")
                .filter(pl.col("writers").is_not_null() & (pl.col("writers") != ""))
                .join(names.select(["nconst", "primaryName"]), left_on="writers", right_on="nconst", how="left")
                .group_by("tconst")
                .agg(pl.col("primaryName").alias("writers"))
            )

        # 2. Resolve cast names (actors/actresses) from principals using names basics
        cast_resolved = None
        if principals is not None and names is not None:
            cast_resolved = (
                principals.filter(pl.col("category").is_in(["actor", "actress", "self"]))
                .join(names.select(["nconst", "primaryName"]), on="nconst", how="left")
                .sort("ordering")
                .group_by("tconst")
                .agg(pl.col("primaryName").alias("cast"))
            )

        # 3. Assemble and join all IMDb tables
        lf = basics
        if ratings is not None:
            lf = lf.join(ratings, on="tconst", how="left")
        if directors_resolved is not None:
            lf = lf.join(directors_resolved, on="tconst", how="left")
        if writers_resolved is not None:
            lf = lf.join(writers_resolved, on="tconst", how="left")
        if cast_resolved is not None:
            lf = lf.join(cast_resolved, on="tconst", how="left")

        # 4. Map columns to canonical representations
        lf_cols = lf.collect_schema().names()
        mapped_lf = lf.select([
            pl.col("tconst").alias("imdb_id"),
            pl.col("primaryTitle").alias("title"),
            pl.col("originalTitle").alias("original_title"),
            pl.col("genres"),
            pl.col("runtimeMinutes").alias("runtime_minutes"),
            pl.col("startYear").alias("release_year"),
            pl.col("averageRating").alias("rating_value") if "averageRating" in lf_cols else pl.lit(None).alias("rating_value"),
            pl.col("numVotes").alias("vote_count") if "numVotes" in lf_cols else pl.lit(None).alias("vote_count"),
            pl.col("cast") if "cast" in lf_cols else pl.lit(None).alias("cast"),
            pl.col("directors") if "directors" in lf_cols else pl.lit(None).alias("directors"),
            pl.col("writers") if "writers" in lf_cols else pl.lit(None).alias("writers")
        ])

        return self.enforce_canonical_schema(mapped_lf, "imdb")


class TMDbAdapter(BaseMovieAdapter):
    """Adapter to map TMDb metadata, credits, and keywords into the canonical schema."""
    def adapt(self, datasets: Dict[str, pl.LazyFrame]) -> pl.LazyFrame:
        logger.info("[TMDbAdapter] Translating tables into canonical schema...")
        meta = datasets.get("movies_metadata.csv")
        credits = datasets.get("credits.csv")
        keywords = datasets.get("keywords.csv")

        if meta is None:
            raise ValueError("TMDb movies_metadata is required for mapping.")

        # 1. Join metadata with credits and keywords
        lf = meta
        if credits is not None:
            lf = lf.join(credits, on="id", how="left")
        if keywords is not None:
            lf = lf.join(keywords, on="id", how="left")

        languages_expr = (
            pl.when(pl.col("original_language").is_not_null())
            .then(pl.concat_list([pl.col("original_language")]))
            .otherwise(pl.lit([]).cast(pl.List(pl.String)))
        )

        
        lf_cols = lf.collect_schema().names()
        mapped_lf = lf.select([
            pl.col("imdb_id"),
            pl.col("id").alias("tmdb_id"),
            pl.col("title"),
            pl.col("original_title"),
            pl.col("overview"),
            pl.col("genres"),
            pl.col("cast_names").alias("cast") if "cast_names" in lf_cols else pl.lit(None).alias("cast"),
            pl.col("director_names").alias("directors") if "director_names" in lf_cols else pl.lit(None).alias("directors"),
            pl.col("runtime").alias("runtime_minutes"),
            pl.col("release_date").dt.year().cast(pl.Int32).alias("release_year"),
            pl.col("vote_average").alias("rating_value"),
            pl.col("vote_count"),
            pl.col("popularity"),
            pl.col("production_companies"),
            languages_expr.alias("languages"),
            pl.col("keywords") if "keywords" in lf_cols else pl.lit(None).alias("keywords")
        ])

        return self.enforce_canonical_schema(mapped_lf, "tmdb")


class MovieLensAdapter(BaseMovieAdapter):
    """Adapter to map MovieLens movies, ratings, and links into the canonical schema."""
    def adapt(self, datasets: Dict[str, pl.LazyFrame]) -> pl.LazyFrame:
        logger.info("[MovieLensAdapter] Translating tables into canonical schema...")
        movies = datasets.get("movies.csv")
        ratings = datasets.get("ratings.csv")
        links = datasets.get("links.csv")

        if movies is None:
            raise ValueError("MovieLens movies table is required for mapping.")

        # 1. Aggregate ratings stats per movie
        ratings_stats = None
        if ratings is not None:
            ratings_stats = ratings.group_by("movieId").agg([
                pl.col("rating").mean().alias("rating_value"),
                pl.col("rating").count().alias("vote_count")
            ])

        # 2. Join tables
        lf = movies
        if links is not None:
            lf = lf.join(links, on="movieId", how="left")
        if ratings_stats is not None:
            lf = lf.join(ratings_stats, on="movieId", how="left")

        # 3. Map columns to canonical schema
        lf_cols = lf.collect_schema().names()
        mapped_lf = lf.select([
            pl.col("imdbId").alias("imdb_id") if "imdbId" in lf_cols else pl.lit(None).alias("imdb_id"),
            pl.col("tmdbId").alias("tmdb_id") if "tmdbId" in lf_cols else pl.lit(None).alias("tmdb_id"),
            pl.col("movieId").alias("movielens_id"),
            pl.col("title"),
            pl.col("genres"),
            pl.col("release_year"),
            pl.col("rating_value") if "rating_value" in lf_cols else pl.lit(None).alias("rating_value"),
            pl.col("vote_count") if "vote_count" in lf_cols else pl.lit(None).alias("vote_count")
        ])

        return self.enforce_canonical_schema(mapped_lf, "movielens")


class WikipediaAdapter(BaseMovieAdapter):
    """Adapter to map Wikipedia plots, casts, and directors into the canonical schema."""
    def adapt(self, datasets: Dict[str, pl.LazyFrame]) -> pl.LazyFrame:
        logger.info("[WikipediaAdapter] Translating tables into canonical schema...")
        plots = datasets.get("wiki_movie_plots_deduped.csv")

        if plots is None:
            raise ValueError("Wikipedia plots table is required for mapping.")

        # 1. Map and parse lists from raw comma/slash separated strings
        # Wikipedia Genre can be separated by commas, slashes, or whitespace
        genres_expr = (
            pl.col("Genre")
            .str.replace_all("/", ",")
            .str.split(",")
            .list.eval(pl.element().str.strip_chars().filter(pl.element().str.strip_chars() != ""))
        )

        directors_expr = (
            pl.col("Director")
            .str.split(",")
            .list.eval(pl.element().str.strip_chars().filter(pl.element().str.strip_chars() != ""))
        )

        cast_expr = (
            pl.col("Cast")
            .str.split(",")
            .list.eval(pl.element().str.strip_chars().filter(pl.element().str.strip_chars() != ""))
        )

        mapped_lf = plots.select([
            pl.col("Wiki Page").alias("wiki_page"),
            pl.col("Title").alias("title"),
            pl.col("Plot").alias("plot_summary"),
            pl.col("Release Year").alias("release_year"),
            genres_expr.alias("genres"),
            directors_expr.alias("directors"),
            cast_expr.alias("cast")
        ])

        return self.enforce_canonical_schema(mapped_lf, "wikipedia")
