import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from loguru import logger
import polars as pl
from app.core.config import settings
from app.datasets.adapters import CANONICAL_COLUMNS



class DatasetMerger:
    """
    Merging engine that combines canonical movie datasets (IMDb, TMDb, MovieLens, Wikipedia)
    into a single unified movie knowledge base.
    """
    def __init__(self, canonical_dir: Optional[str] = None, output_dir: Optional[str] = None) -> None:
        self.canonical_dir = Path(canonical_dir or settings.PROCESSED_DATA_DIR) / "canonical"
        self.output_dir = Path(output_dir or settings.PROCESSED_DATA_DIR) / "canonical"
        logger.info(f"DatasetMerger initialized. Canonical dir: {self.canonical_dir}")

    def run_merging(self) -> Dict[str, Any]:
        """Runs the linking graph construction, catalog filtering, and consolidation joins."""
        logger.info("Executing dataset merging engine...")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load canonical datasets lazily
        imdb = pl.scan_parquet(self.canonical_dir / "imdb_canonical.parquet")
        tmdb = pl.scan_parquet(self.canonical_dir / "tmdb_canonical.parquet")
        movielens = pl.scan_parquet(self.canonical_dir / "movielens_canonical.parquet")
        wikipedia = pl.scan_parquet(self.canonical_dir / "wikipedia_canonical.parquet")

        # 2. Ingest linked IDs to filter the IMDb catalog
        logger.info("[Merger] Resolving cross-dataset links...")
        tmdb_linked = tmdb.filter(pl.col("imdb_id").is_not_null()).select("imdb_id").collect().to_series().to_list()
        ml_linked = movielens.filter(pl.col("imdb_id").is_not_null()).select("imdb_id").collect().to_series().to_list()
        linked_imdb_ids = list(set(tmdb_linked + ml_linked))
        logger.info(f"[Merger] Found {len(linked_imdb_ids):,} linked IMDb IDs from TMDb & MovieLens.")

        # Filter IMDb to keep:
        # - Movies linked to TMDb/MovieLens
        # - OR actual movies with rating_value and vote_count >= 50 (discarding game/episode/pilot noise)
        imdb_filtered = imdb.filter(
            pl.col("imdb_id").is_in(linked_imdb_ids) |
            (pl.col("rating_value").is_not_null() & (pl.col("vote_count") >= 50))
        )

        # 3. Create Title-Year Lookup from filtered IMDb for Wikipedia matching
        logger.info("[Merger] Creating title-year lookup for Wikipedia plots...")
        imdb_lookup = imdb_filtered.select(["imdb_id", "title", "release_year"]).collect()
        imdb_lookup = imdb_lookup.with_columns(
            pl.col("title").str.to_lowercase().str.replace_all(r"[^a-z0-9]", "").alias("norm_title")
        )

        # 4. Resolve Wikipedia links using title+year matching
        logger.info("[Merger] Aligning Wikipedia plots on title & release year...")
        wiki_df = wikipedia.collect().drop(["imdb_id", "tmdb_id", "movielens_id"]).with_columns(
            pl.col("title").str.to_lowercase().str.replace_all(r"[^a-z0-9]", "").alias("norm_title")
        )
        wiki_linked = wiki_df.join(
            imdb_lookup,
            left_on=["norm_title", "release_year"],
            right_on=["norm_title", "release_year"],
            how="left"
        )

        
        wiki_with_imdb = wiki_linked.filter(pl.col("imdb_id").is_not_null()).select(["wiki_page", "imdb_id"])
        wiki_unlinked = wiki_linked.filter(pl.col("imdb_id").is_null()).select(["wiki_page"])
        logger.info(
            f"[Merger] Wikipedia linking stats: matched {wiki_with_imdb.height:,} plots to IMDb IDs. "
            f"Left {wiki_unlinked.height:,} plots unlinked."
        )

        # 5. Build Unified ID Map (imdb_id, tmdb_id, movielens_id, wiki_page)
        logger.info("[Merger] Building master movie linkage map...")
        
        # Base links from MovieLens and TMDb metadata
        ml_links = movielens.select(["imdb_id", "tmdb_id", "movielens_id"]).collect()
        tmdb_links = tmdb.select(["imdb_id", "tmdb_id"]).collect().with_columns(pl.lit(None).cast(pl.Int64).alias("movielens_id"))
        
        links_combined = pl.concat([ml_links, tmdb_links]).unique(subset=["imdb_id", "tmdb_id"])
        
        # Deduplicate to single mapping per IMDb ID
        master_links = links_combined.group_by("imdb_id").agg([
            pl.col("tmdb_id").drop_nulls().first(),
            pl.col("movielens_id").drop_nulls().first()
        ]).filter(pl.col("imdb_id").is_not_null())
        
        # Join Wikipedia links on IMDb ID
        master_links = master_links.join(wiki_with_imdb, on="imdb_id", how="left")
        
        # Append TMDb entries that don't have an IMDb ID
        tmdb_only = tmdb_links.filter(pl.col("imdb_id").is_null() & pl.col("tmdb_id").is_not_null()).group_by("tmdb_id").agg([
            pl.col("imdb_id").first(), # will be null
            pl.col("movielens_id").first() # will be null
        ]).with_columns(pl.lit(None).cast(pl.String).alias("wiki_page"))
        
        # Append Wikipedia entries that are unlinked
        wiki_only = wiki_unlinked.with_columns([
            pl.lit(None).cast(pl.String).alias("imdb_id"),
            pl.lit(None).cast(pl.Int64).alias("tmdb_id"),
            pl.lit(None).cast(pl.Int64).alias("movielens_id")
        ])

        # Append remaining IMDb-only entries (movies with 50+ votes not linked to TMDb/ML/Wiki)
        imdb_lookup_ids = imdb_lookup.select("imdb_id").unique()
        imdb_only = imdb_lookup_ids.join(master_links, on="imdb_id", how="anti").with_columns([
            pl.lit(None).cast(pl.Int64).alias("tmdb_id"),
            pl.lit(None).cast(pl.Int64).alias("movielens_id"),
            pl.lit(None).cast(pl.String).alias("wiki_page")
        ])

        # Combine all sets into a master linkage table
        master_map = pl.concat([
            master_links.select(["imdb_id", "tmdb_id", "movielens_id", "wiki_page"]),
            tmdb_only.select(["imdb_id", "tmdb_id", "movielens_id", "wiki_page"]),
            wiki_only.select(["imdb_id", "tmdb_id", "movielens_id", "wiki_page"]),
            imdb_only.select(["imdb_id", "tmdb_id", "movielens_id", "wiki_page"])
        ]).unique(subset=["imdb_id", "tmdb_id", "wiki_page"]).with_row_index("movie_idx")

        logger.info(f"[Merger] Created Master Link Map containing {master_map.height:,} unique movies.")

        # 6. Associate movie_idx to all datasets
        logger.info("[Merger] Re-keying datasets with canonical movie indices...")
        master_map_lazy = master_map.lazy()
        
        imdb_keyed = imdb_filtered.join(master_map_lazy.select(["movie_idx", "imdb_id"]), on="imdb_id", how="inner")
        tmdb_keyed = tmdb.join(master_map_lazy.select(["movie_idx", "tmdb_id"]), on="tmdb_id", how="inner")
        movielens_keyed = movielens.join(master_map_lazy.select(["movie_idx", "movielens_id"]), on="movielens_id", how="inner")
        wiki_keyed = wikipedia.join(master_map_lazy.select(["movie_idx", "wiki_page"]), on="wiki_page", how="inner")

        # 7. Concatenate and Sort by dataset priority
        # TMDb (1) -> IMDb (2) -> MovieLens (3) -> Wikipedia (4)
        logger.info("[Merger] Concatenating records and resolving feature conflicts...")
        all_rows = pl.concat([
            imdb_keyed,
            tmdb_keyed,
            movielens_keyed,
            wiki_keyed
        ])

        # Create source dataset priority column
        all_rows = all_rows.with_columns(
            pl.when(pl.col("source_dataset") == "tmdb").then(1)
            .when(pl.col("source_dataset") == "imdb").then(2)
            .when(pl.col("source_dataset") == "movielens").then(3)
            .otherwise(4)
            .alias("source_priority")
        ).sort("source_priority")

        # 8. Group by movie_idx and aggregate columns
        # - Single-value fields: take the first non-null value (automatically selects higher priority source)
        # - List fields: explode, unique, and drop nulls
        # - Ratings: compute weighted ratings using vote counts
        weighted_rating_expr = (
            pl.when(pl.col("vote_count").drop_nulls().sum() > 0)
            .then(
                (pl.col("rating_value") * pl.col("vote_count")).drop_nulls().sum() /
                pl.col("vote_count").drop_nulls().sum()
            )
            .otherwise(pl.col("rating_value").drop_nulls().first())
        )

        logger.info("[Merger] Running group-by aggregation to build final movie knowledge base...")
        final_movies = all_rows.group_by("movie_idx").agg([
            pl.col("imdb_id").drop_nulls().first(),
            pl.col("tmdb_id").drop_nulls().first(),
            pl.col("movielens_id").drop_nulls().first(),
            pl.col("wiki_page").drop_nulls().first(),
            pl.col("title").drop_nulls().first(),
            pl.col("original_title").drop_nulls().first(),
            pl.col("overview").drop_nulls().first(),
            pl.col("plot_summary").drop_nulls().first(),
            pl.col("genres").list.explode().unique().drop_nulls(),
            pl.col("cast").list.explode().unique().drop_nulls(),
            pl.col("directors").list.explode().unique().drop_nulls(),
            pl.col("writers").list.explode().unique().drop_nulls(),
            pl.col("runtime_minutes").drop_nulls().first(),
            pl.col("release_year").drop_nulls().first(),
            weighted_rating_expr.alias("rating_value"),
            pl.col("vote_count").drop_nulls().sum(),
            pl.col("popularity").drop_nulls().first(),
            pl.col("production_companies").list.explode().unique().drop_nulls(),
            pl.col("languages").list.explode().unique().drop_nulls(),
            pl.col("keywords").list.explode().unique().drop_nulls(),
            pl.col("source_dataset").drop_nulls().first(),

            pl.col("poster_path").drop_nulls().first(),
            pl.col("backdrop_path").drop_nulls().first(),
            pl.col("trailer_url").drop_nulls().first(),
            pl.col("streaming_providers").list.explode().unique().drop_nulls(),
            pl.col("collection_name").drop_nulls().first(),
            pl.col("certification").drop_nulls().first(),
            pl.col("document").drop_nulls().first()
        ]).drop("movie_idx").collect()

        # Generate natural language summaries for embedding
        from app.datasets.document_generator import generate_knowledge_base_documents
        logger.info("[Merger] Generating natural-language movie documents for embedding...")
        final_movies = generate_knowledge_base_documents(final_movies)

        # Enforce canonical columns ordering
        final_movies = final_movies.select(CANONICAL_COLUMNS)



        output_path = self.output_dir / "movies_knowledge_base.parquet"
        logger.info(f"[Merger] Writing unified knowledge base to {output_path}...")
        final_movies.write_parquet(output_path)

        # 9. Compute overlap statistics for summary
        logger.info("[Merger] Computing overlap statistics...")
        stats = {
            "total_canonical_movies": final_movies.height,
            "imdb_source_rows": imdb_filtered.select(pl.len()).collect().item(),
            "tmdb_source_rows": tmdb.select(pl.len()).collect().item(),
            "movielens_source_rows": movielens.select(pl.len()).collect().item(),
            "wikipedia_source_rows": wikipedia.select(pl.len()).collect().item(),
            "overlap_imdb_tmdb": master_map.filter(pl.col("imdb_id").is_not_null() & pl.col("tmdb_id").is_not_null()).height,
            "overlap_imdb_movielens": master_map.filter(pl.col("imdb_id").is_not_null() & pl.col("movielens_id").is_not_null()).height,
            "overlap_imdb_wiki": master_map.filter(pl.col("imdb_id").is_not_null() & pl.col("wiki_page").is_not_null()).height,
            "movies_with_ratings": final_movies.filter(pl.col("rating_value").is_not_null()).height,
            "movies_with_plots": final_movies.filter(pl.col("plot_summary").is_not_null() | pl.col("overview").is_not_null()).height
        }

        logger.info(
            f"[Merger] Merging complete. Generated {stats['total_canonical_movies']:,} canonical movies. "
            f"Movies with ratings: {stats['movies_with_ratings']:,}. "
            f"Movies with plot descriptions: {stats['movies_with_plots']:,}."
        )

        return stats
