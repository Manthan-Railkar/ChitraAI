"""
Qdrant Ingestion Pipeline for ChitraAI.

Reads consolidated movie knowledge base metadata and generated vector embeddings,
creates the Qdrant movie collection if not exists, and uploads the data
in batches with retries, checkpointing, and duplicate prevention.
"""

import os
import sys
import json
import uuid
import time
import argparse
from pathlib import Path
from typing import Optional, List, Set

import polars as pl
from loguru import logger
from qdrant_client.models import PointStruct

# Add the backend directory to sys.path to allow imports from app
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.vector_db.qdrant import QdrantWrapper


def generate_movie_uuid(
    imdb_id: Optional[str],
    tmdb_id: Optional[int],
    title: str,
    release_year: Optional[int]
) -> str:
    """
    Generates a deterministic UUID v5 for a movie based on its unique identifiers.
    Ensures duplicate prevention and idempotent upserts in Qdrant.
    """
    parts = []
    if imdb_id and imdb_id.strip():
        parts.append(f"imdb:{imdb_id.strip()}")
    if tmdb_id:
        parts.append(f"tmdb:{tmdb_id}")
    
    # Fallback if standard IDs are missing
    if not parts:
        clean_title = "".join(char for char in title.lower() if char.isalnum())
        parts.append(f"title_year:{clean_title}_{release_year or 0}")

    unique_str = "|".join(parts)
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))


def _get_checkpoint_path() -> Path:
    """Returns the path to the Qdrant ingestion checkpoint JSON file."""
    checkpoint_dir = Path(settings.EMBEDDINGS_DIR) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / "qdrant_ingest_checkpoint.json"


def _load_ingest_checkpoint() -> Set[str]:
    """Loads already ingested point UUIDs from the local checkpoint file."""
    path = _get_checkpoint_path()
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"Loaded checkpoint with {len(data):,} already ingested movie UUIDs.")
                return set(data)
        except Exception as e:
            logger.warning(f"Failed to load ingestion checkpoint: {e}. Starting fresh.")
            return set()
    return set()


def _save_ingest_checkpoint(ingested_uuids: Set[str]) -> None:
    """Saves the current set of ingested point UUIDs to the local checkpoint file."""
    path = _get_checkpoint_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(list(ingested_uuids), f)
        logger.debug(f"Saved ingestion checkpoint with {len(ingested_uuids):,} UUIDs.")
    except Exception as e:
        logger.error(f"Failed to save ingestion checkpoint: {e}")


def _load_and_join_datasets(limit: Optional[int] = None) -> pl.DataFrame:
    """
    Loads embeddings and joins them with the canonical movie metadata.
    """
    embeddings_path = Path(settings.EMBEDDINGS_DIR) / "movie_embeddings.parquet"
    kb_path = Path(settings.MERGED_DATA_DIR) / "movies_knowledge_base.parquet"

    # Fallback to processed canonical if merged doesn't exist
    if not kb_path.exists():
        kb_path = Path(settings.PROCESSED_DATA_DIR) / "canonical" / "movies_knowledge_base.parquet"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings Parquet file not found at {embeddings_path}. Run build_embeddings.py first.")
    if not kb_path.exists():
        raise FileNotFoundError(f"Canonical movie knowledge base Parquet not found at {kb_path}. Run preprocess_datasets first.")

    logger.info(f"Loading embeddings from {embeddings_path}...")
    embeddings_df = pl.read_parquet(embeddings_path)
    logger.info(f"Loaded {embeddings_df.height:,} embedding records.")

    if limit is not None and limit > 0:
        embeddings_df = embeddings_df.head(limit)
        logger.info(f"Limited to first {limit:,} records for ingestion.")

    logger.info(f"Loading canonical metadata from {kb_path}...")
    # Load all canonical columns except 'document' if we want to save space, but doc is requested so we load all
    kb_df = pl.read_parquet(kb_path)

    # Define deterministic movie_key expression to handle nulls in ID fields during join
    key_expr = (
        pl.when(pl.col("imdb_id").is_not_null())
        .then(pl.lit("imdb:") + pl.col("imdb_id"))
        .otherwise(
            pl.when(pl.col("tmdb_id").is_not_null())
            .then(pl.lit("tmdb:") + pl.col("tmdb_id").cast(pl.String))
            .otherwise(
                pl.lit("title_year:") + pl.col("title").str.to_lowercase() + pl.lit("_") + pl.col("release_year").cast(pl.String).fill_null("0")
            )
        )
    ).alias("movie_key")

    embeddings_df = embeddings_df.with_columns(key_expr)
    kb_df = kb_df.with_columns(key_expr)

    # Perform inner join to align metadata with generated embeddings on movie_key
    logger.info("Aligning embeddings and metadata via inner join on movie_key...")
    df = embeddings_df.join(kb_df.drop(["imdb_id", "tmdb_id", "movielens_id", "title", "release_year"]), on="movie_key", how="inner")
    
    # Drop temp movie_key column
    df = df.drop("movie_key")
    
    logger.info(f"Successfully joined dataset: {df.height:,} rows aligned.")

    return df



def ingest_movies_to_qdrant(
    limit: Optional[int] = None,
    batch_size: Optional[int] = None,
    fresh: bool = False,
    resume: bool = True
) -> dict:
    """
    Ingestion pipeline main execution function.
    """
    setup_logging()
    logger.info("=" * 60)
    logger.info("Starting ChitraAI Qdrant Ingestion Pipeline")
    logger.info("=" * 60)

    start_time = time.time()
    batch_size = batch_size or settings.QDRANT_BATCH_SIZE
    checkpoint_interval = settings.QDRANT_INGEST_CHECKPOINT_INTERVAL

    # 1. Connect to Qdrant client
    qdrant_wrapper = QdrantWrapper()
    if not qdrant_wrapper.connect():
        logger.error("Could not establish Qdrant connection. Ingestion aborted.")
        sys.exit(1)

    # 2. Check for fresh run or clean ups
    if fresh:
        logger.info("Fresh start requested. Clearing existing Qdrant collection and checkpoints.")
        if qdrant_wrapper.collection_exists():
            qdrant_wrapper.client.delete_collection(qdrant_wrapper.collection_name)
            logger.info(f"Deleted collection '{qdrant_wrapper.collection_name}'.")
        
        checkpoint_path = _get_checkpoint_path()
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("Deleted local ingestion checkpoint file.")

    # 3. Load and join datasets
    df = _load_and_join_datasets(limit=limit)
    if df.height == 0:
        logger.warning("No aligned records to ingest. Pipeline finished.")
        return {"status": "no_data", "ingested_count": 0}

    # Extract dimension from the first embedding to configure the collection
    sample_emb = df.select("embedding").row(0)[0]
    vector_dim = len(sample_emb) if sample_emb else 768

    # 4. Create collection if not exists
    if not qdrant_wrapper.create_collection(vector_size=vector_dim):
        logger.error("Failed to verify/create Qdrant collection. Ingestion aborted.")
        sys.exit(1)

    # 5. Load checkpoints
    ingested_uuids = set() if (fresh or not resume) else _load_ingest_checkpoint()

    # 6. Map rows to Qdrant PointStruct and filter duplicates
    logger.info("Preparing data payloads and generating point UUIDs...")
    
    # Select columns to convert to payload dictionary
    payload_cols = [col for col in df.columns if col != "embedding"]

    # Convert to python dicts for mapping
    rows = df.to_dicts()
    
    points_to_upload: List[PointStruct] = []
    skipped_duplicates = 0
    skipped_checkpoints = 0

    for row in rows:
        imdb_id = row.get("imdb_id")
        tmdb_id = row.get("tmdb_id")
        title = row.get("title", "")
        release_year = row.get("release_year")
        embedding = row.get("embedding")

        if not embedding or len(embedding) != vector_dim:
            logger.warning(f"Skipping movie '{title}' due to missing or invalid embedding dimensions.")
            continue

        point_uuid = generate_movie_uuid(imdb_id, tmdb_id, title, release_year)

        # Duplicate prevention (local checkpoint skip)
        if point_uuid in ingested_uuids:
            skipped_checkpoints += 1
            continue

        # Build payload dict (serialize lists cleanly, clean NaNs/nulls)
        payload = {}
        for col in payload_cols:
            val = row.get(col)
            # Polars conversion handles lists and primitives well. Replace NaN with None.
            if isinstance(val, float) and (val != val):  # check for NaN
                val = None
            elif isinstance(val, pl.Series):
                val = val.to_list()
            payload[col] = val

        # Clean list-like fields inside payload to normal python lists
        for k, v in payload.items():
            if isinstance(v, list):
                # Ensure elements are clean
                payload[k] = [el for el in v if el is not None]

        points_to_upload.append(
            PointStruct(
                id=point_uuid,
                vector=[float(x) for x in embedding],
                payload=payload
            )
        )

    total_to_ingest = len(points_to_upload)
    logger.info(
        f"Ingestion queue summary: "
        f"Total aligned={len(rows):,} | "
        f"Already ingested={skipped_checkpoints:,} | "
        f"Queue size={total_to_ingest:,}"
    )

    if total_to_ingest == 0:
        logger.info("All records already ingested. Verification completed.")
        return {
            "status": "complete",
            "ingested_count": 0,
            "total_indexed": qdrant_wrapper.count_points()
        }

    # 7. Ingestion batch loop
    logger.info(f"Ingesting {total_to_ingest:,} points in batches of {batch_size}...")
    num_batches = (total_to_ingest + batch_size - 1) // batch_size
    newly_ingested = 0
    batch_uuids_buffer: List[str] = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_to_ingest)
        batch = points_to_upload[start_idx:end_idx]

        # Upload batch
        if qdrant_wrapper.upload_batch(batch):
            newly_ingested += len(batch)
            batch_uuids_buffer.extend([pt.id for pt in batch])
            
            # Periodically write checkpoints
            if (batch_idx + 1) % checkpoint_interval == 0 or (batch_idx + 1) == num_batches:
                ingested_uuids.update(batch_uuids_buffer)
                _save_ingest_checkpoint(ingested_uuids)
                batch_uuids_buffer.clear()

            progress = (newly_ingested / total_to_ingest) * 100
            logger.info(
                f"Batch {batch_idx + 1}/{num_batches} | "
                f"Ingested {newly_ingested:,}/{total_to_ingest:,} ({progress:.1f}%)"
            )
        else:
            # Save progress before exit
            if batch_uuids_buffer:
                ingested_uuids.update(batch_uuids_buffer)
                _save_ingest_checkpoint(ingested_uuids)
            logger.error(f"Failed to ingest batch {batch_idx + 1}. Ingestion halted.")
            sys.exit(1)

    # 8. Post-ingestion verification
    logger.info("Verifying point counts from Qdrant...")
    qdrant_count = qdrant_wrapper.count_points()
    elapsed_time = time.time() - start_time
    logger.info(f"Indexed points count in Qdrant collection '{qdrant_wrapper.collection_name}': {qdrant_count:,}")

    logger.info("=" * 60)
    logger.info("Qdrant Ingestion Pipeline completed successfully!")
    logger.info(f"Newly Ingested: {newly_ingested:,} | Total indexed in Qdrant: {qdrant_count:,} | Time: {elapsed_time/60:.2f} min")
    logger.info("=" * 60)

    # Save final execution summary
    summary = {
        "status": "success",
        "newly_ingested": newly_ingested,
        "total_indexed": qdrant_count,
        "elapsed_time_seconds": round(elapsed_time, 2),
        "collection_name": qdrant_wrapper.collection_name,
        "is_local_client": qdrant_wrapper._is_local,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    
    report_path = Path(settings.REPORTS_DIR) / "qdrant_ingest_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved ingestion summary report to {report_path}")

    return summary


def main():
    """CLI entry point for the Qdrant ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="ChitraAI Qdrant Ingestion Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of movies to ingest."
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override default batch size."
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Clear existing Qdrant collection and checkpoints before starting."
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore checkpoints and re-upload all points."
    )

    args = parser.parse_args()

    ingest_movies_to_qdrant(
        limit=args.limit,
        batch_size=args.batch_size,
        fresh=args.fresh,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
