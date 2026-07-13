import os
import json
import time
from pathlib import Path
from typing import Dict, Any

from loguru import logger
import polars as pl
from app.core.config import settings
from app.datasets.adapters import CANONICAL_SCHEMA, CANONICAL_COLUMNS


def run_qa_and_export() -> Dict[str, Any]:
    """
    Runs final quality assurance validation checks on the movie knowledge base,
    compiles detailed statistics & distributions, and exports the clean data
    in Parquet, CSV, and JSON formats.
    """
    logger.info("Starting Quality Assurance and Export Pipeline...")

    canonical_kb_path = Path(settings.PROCESSED_DATA_DIR) / "canonical" / "movies_knowledge_base.parquet"
    if not canonical_kb_path.exists():
        raise FileNotFoundError(f"Merged movies knowledge base not found at {canonical_kb_path}")

    # 1. Load consolidated knowledge base
    logger.info(f"Loading merged knowledge base from {canonical_kb_path}...")
    df = pl.read_parquet(canonical_kb_path)
    initial_row_count = df.height

    # 2. Perform validation checks
    logger.info("Executing validation checks...")

    # Duplicate movie checks
    dup_imdb = df.filter(pl.col("imdb_id").is_not_null()).filter(pl.col("imdb_id").is_duplicated()).height
    dup_tmdb = df.filter(pl.col("tmdb_id").is_not_null()).filter(pl.col("tmdb_id").is_duplicated()).height
    dup_title_year = df.filter(pl.struct(["title", "release_year"]).is_duplicated()).height

    # Missing/empty columns
    missing_titles = df.filter(pl.col("title").is_null() | (pl.col("title").str.strip_chars() == "")).height
    empty_overviews = df.filter(
        (pl.col("overview").is_null() | (pl.col("overview").str.strip_chars() == "")) &
        (pl.col("plot_summary").is_null() | (pl.col("plot_summary").str.strip_chars() == ""))
    ).height
    missing_genres = df.filter(pl.col("genres").is_null() | (pl.col("genres").list.len() == 0)).height

    # Invalid identifiers
    invalid_imdb_format = df.filter(
        pl.col("imdb_id").is_not_null() & ~pl.col("imdb_id").str.contains(r"^tt\d{7,8}$")
    ).height
    invalid_tmdb = df.filter(pl.col("tmdb_id").is_not_null() & (pl.col("tmdb_id") <= 0)).height
    invalid_movielens = df.filter(pl.col("movielens_id").is_not_null() & (pl.col("movielens_id") <= 0)).height

    # Rating anomalies
    invalid_ratings = df.filter(
        pl.col("rating_value").is_not_null() & ((pl.col("rating_value") < 1.0) | (pl.col("rating_value") > 10.0))
    ).height

    # Orphan records
    orphans = df.filter(
        pl.col("imdb_id").is_null() &
        pl.col("tmdb_id").is_null() &
        pl.col("movielens_id").is_null() &
        pl.col("wiki_page").is_null()
    ).height

    # Schema consistency
    schema_keys_match = list(df.schema.keys()) == CANONICAL_COLUMNS
    schema_types_match = all(df.schema[col] == CANONICAL_SCHEMA[col] for col in CANONICAL_COLUMNS)
    schema_consistent = schema_keys_match and schema_types_match

    # 3. Calculate distributions
    logger.info("Computing metadata distributions...")

    # Genres distribution
    genre_counts = (
        df.select("genres")
        .explode("genres", empty_as_null=True)
        .filter(pl.col("genres").is_not_null())
        .group_by("genres")
        .len()
        .sort("len", descending=True)
    )
    genres_dist = {row["genres"]: row["len"] for row in genre_counts.iter_rows(named=True)}

    # Languages distribution (top 30)
    lang_counts = (
        df.select("languages")
        .explode("languages", empty_as_null=True)
        .filter(pl.col("languages").is_not_null())
        .group_by("languages")
        .len()
        .sort("len", descending=True)
        .head(30)
    )
    languages_dist = {row["languages"]: row["len"] for row in lang_counts.iter_rows(named=True)}

    # Rating distribution in bins: [1.0-2.0), [2.0-3.0), ..., [9.0-10.0]
    rating_bins = (
        df.filter(pl.col("rating_value").is_not_null())
        .select(
            pl.col("rating_value")
            .floor()
            .clip(lower_bound=1.0, upper_bound=9.0)
            .cast(pl.Int32)
            .alias("bin")
        )
        .group_by("bin")
        .len()
        .sort("bin")
    )
    ratings_dist = {f"{row['bin']}.0-{row['bin']+1}.0": row["len"] for row in rating_bins.iter_rows(named=True)}

    # Missingness summary per column
    missing_summary = {}
    for col in CANONICAL_COLUMNS:
        null_count = df.select(pl.col(col).is_null().sum()).item()
        missing_summary[col] = {
            "null_count": null_count,
            "null_percentage": round((null_count / initial_row_count) * 100, 3)
        }

    # 4. Filter and finalize dataset
    logger.info("Filtering invalid records...")
    # Clean rules: must have title, cannot be orphan, rating must be valid if present
    clean_df = df.filter(
        pl.col("title").is_not_null() & (pl.col("title").str.strip_chars() != "") &
        ~(
            pl.col("imdb_id").is_null() &
            pl.col("tmdb_id").is_null() &
            pl.col("movielens_id").is_null() &
            pl.col("wiki_page").is_null()
        ) &
        (pl.col("rating_value").is_null() | ((pl.col("rating_value") >= 1.0) & (pl.col("rating_value") <= 10.0)))
    )
    final_row_count = clean_df.height
    removed_records_count = initial_row_count - final_row_count

    # 5. Compile QA Report
    qa_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_statistics": {
            "initial_rows": initial_row_count,
            "final_rows": final_row_count,
            "removed_rows": removed_records_count,
            "column_count": len(CANONICAL_COLUMNS),
            "schema_consistent": schema_consistent
        },
        "quality_metrics": {
            "duplicate_imdb_ids": dup_imdb,
            "duplicate_tmdb_ids": dup_tmdb,
            "duplicate_title_year_combinations": dup_title_year,
            "missing_titles": missing_titles,
            "empty_overviews": empty_overviews,
            "missing_genres": missing_genres,
            "invalid_imdb_format": invalid_imdb_format,
            "invalid_tmdb_ids": invalid_tmdb,
            "invalid_movielens_ids": invalid_movielens,
            "invalid_ratings": invalid_ratings,
            "orphan_records": orphans
        },
        "distributions": {
            "genres": genres_dist,
            "languages": languages_dist,
            "ratings": ratings_dist
        },
        "missing_value_summary": missing_summary
    }

    # Save QA Report
    reports_dir = Path(settings.REPORTS_DIR)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "movies_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(qa_report, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved detailed QA validation report to {report_path}")

    # 6. Export to Target Formats
    merged_dir = Path(settings.MERGED_DATA_DIR)
    merged_dir.mkdir(parents=True, exist_ok=True)

    # Parquet Export (native support for nested types)
    parquet_path = merged_dir / "movies_knowledge_base.parquet"
    logger.info(f"Exporting final knowledge base to Parquet: {parquet_path}...")
    clean_df.write_parquet(parquet_path)

    # CSV Export – serialize List columns to JSON strings (CSV has no nested type support)
    csv_path = merged_dir / "movies_knowledge_base.csv"
    logger.info(f"Exporting final knowledge base to CSV: {csv_path}...")
    list_cols = [col for col, dtype in clean_df.schema.items() if dtype.base_type() == pl.List]
    csv_df = clean_df.with_columns(
        [pl.col(c).list.join("|").alias(c) for c in list_cols]
    )
    csv_df.write_csv(csv_path)

    # JSON Export (newline-delimited JSON for streaming compatibility)
    json_path = merged_dir / "movies_knowledge_base.json"
    logger.info(f"Exporting final knowledge base to JSON: {json_path}...")
    clean_df.write_ndjson(json_path)

    logger.info("Quality Assurance and Export Pipeline completed successfully.")
    return qa_report


if __name__ == "__main__":
    run_qa_and_export()

