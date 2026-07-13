"""
Embedding Generation Pipeline for ChitraAI.

Loads the canonical movie knowledge base, generates dense vector embeddings
for every movie document using a pretrained Sentence Transformer model,
and persists the results with checkpointing and error recovery.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from loguru import logger

# Add the backend directory to sys.path to allow imports from app
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.services.embedding_service import EmbeddingService


# Columns to carry alongside embeddings for downstream linkage
METADATA_COLUMNS = ["imdb_id", "tmdb_id", "movielens_id", "title", "release_year"]


def _load_knowledge_base(limit: Optional[int] = None) -> pl.DataFrame:
    """
    Loads the canonical movie knowledge base and filters to rows with valid documents.
    
    Args:
        limit: Optional cap on the number of rows to process.
        
    Returns:
        DataFrame with metadata columns + 'document' column.
    """
    kb_path = Path(settings.PROCESSED_DATA_DIR) / "canonical" / "movies_knowledge_base.parquet"
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base not found at {kb_path}")

    logger.info(f"Loading knowledge base from {kb_path}...")
    columns = METADATA_COLUMNS + ["document"]
    df = pl.read_parquet(kb_path, columns=columns)

    # Filter out rows with null or empty documents
    initial_count = df.height
    df = df.filter(
        pl.col("document").is_not_null() & (pl.col("document").str.strip_chars() != "")
    )
    filtered_count = initial_count - df.height
    if filtered_count > 0:
        logger.warning(f"Filtered out {filtered_count:,} rows with null/empty documents.")

    if limit is not None and limit > 0:
        df = df.head(limit)
        logger.info(f"Limited to first {limit:,} movies for processing.")

    logger.info(f"Loaded {df.height:,} movies with valid documents.")
    return df


def _get_checkpoint_path() -> Path:
    """Returns the path to the checkpoint Parquet file."""
    checkpoint_dir = Path(settings.EMBEDDINGS_DIR) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / "embedding_checkpoint.parquet"


def _load_checkpoint() -> Optional[pl.DataFrame]:
    """
    Loads an existing checkpoint if one exists.
    
    Returns:
        DataFrame with previously computed embeddings, or None if no checkpoint exists.
    """
    checkpoint_path = _get_checkpoint_path()
    if checkpoint_path.exists():
        try:
            df = pl.read_parquet(checkpoint_path)
            logger.info(f"Loaded checkpoint with {df.height:,} previously embedded movies.")
            return df
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
            return None
    return None


def _save_checkpoint(df: pl.DataFrame) -> None:
    """Saves the current embedding state to the checkpoint Parquet file."""
    checkpoint_path = _get_checkpoint_path()
    df.write_parquet(checkpoint_path)
    logger.debug(f"Checkpoint saved: {df.height:,} embeddings at {checkpoint_path.name}")


def _filter_already_processed(
    full_df: pl.DataFrame, checkpoint_df: Optional[pl.DataFrame]
) -> pl.DataFrame:
    """
    Removes rows that have already been processed (present in checkpoint).
    Uses imdb_id as the primary key, falling back to title+release_year for
    movies without IMDb IDs.
    """
    if checkpoint_df is None or checkpoint_df.height == 0:
        return full_df

    # Build a set of already-processed identifiers
    processed_imdb = set(
        checkpoint_df.filter(pl.col("imdb_id").is_not_null())
        .select("imdb_id")
        .to_series()
        .to_list()
    )

    # Anti-join: keep rows NOT already in the checkpoint
    if len(processed_imdb) > 0:
        remaining = full_df.filter(
            pl.col("imdb_id").is_null() | ~pl.col("imdb_id").is_in(list(processed_imdb))
        )
    else:
        remaining = full_df

    skipped = full_df.height - remaining.height
    if skipped > 0:
        logger.info(f"Skipping {skipped:,} already-embedded movies from checkpoint.")

    return remaining


def _build_embedding_rows(
    batch_df: pl.DataFrame,
    embeddings: np.ndarray,
) -> pl.DataFrame:
    """
    Combines metadata columns with embedding vectors into a single DataFrame.
    Embeddings are stored as List(Float32) for Parquet compatibility.
    """
    # Convert numpy embeddings to list of lists for Polars
    embedding_lists = [emb.tolist() for emb in embeddings]

    result = batch_df.select(METADATA_COLUMNS).with_columns(
        pl.Series("embedding", embedding_lists, dtype=pl.List(pl.Float32))
    )
    return result


def build_movie_embeddings(
    limit: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    resume: bool = True,
    fresh: bool = False,
) -> dict:
    """
    Main embedding generation pipeline.
    
    Args:
        limit: Optional cap on number of movies to process.
        batch_size: Override for EMBEDDING_BATCH_SIZE setting.
        device: Override for DEVICE setting ('cpu' or 'cuda').
        resume: If True, resume from the latest checkpoint (default).
        fresh: If True, ignore existing checkpoints and start from scratch.
        
    Returns:
        Dictionary with pipeline summary statistics.
    """
    setup_logging()
    logger.info("=" * 60)
    logger.info("Starting ChitraAI Embedding Generation Pipeline")
    logger.info("=" * 60)

    pipeline_start = time.time()
    batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
    checkpoint_interval = settings.EMBEDDING_CHECKPOINT_INTERVAL

    # Ensure output directories exist
    embeddings_dir = Path(settings.EMBEDDINGS_DIR)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(settings.REPORTS_DIR)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load knowledge base
    full_df = _load_knowledge_base(limit=limit)

    # 2. Handle checkpointing
    checkpoint_df = None
    if fresh:
        logger.info("Fresh start requested. Ignoring existing checkpoints.")
        checkpoint_path = _get_checkpoint_path()
        if checkpoint_path.exists():
            checkpoint_path.unlink()
    elif resume:
        checkpoint_df = _load_checkpoint()

    # 3. Filter already-processed movies
    remaining_df = _filter_already_processed(full_df, checkpoint_df)

    if remaining_df.height == 0:
        logger.info("All movies have already been embedded. Nothing to do.")
        # Still produce final output from checkpoint
        if checkpoint_df is not None:
            _export_final(checkpoint_df, embeddings_dir, reports_dir, pipeline_start, device)
        return {"status": "complete", "newly_embedded": 0}

    total_remaining = remaining_df.height
    logger.info(f"Movies to embed: {total_remaining:,} (batch size: {batch_size})")

    # 4. Initialize embedding service
    embedding_service = EmbeddingService(device=device)
    embedding_dim = embedding_service.get_embedding_dimension()
    resolved_device = embedding_service.device
    logger.info(f"Embedding model ready: dim={embedding_dim}, device={resolved_device}")

    # 5. Batched encoding with progress tracking
    num_batches = (total_remaining + batch_size - 1) // batch_size
    accumulated_results: list[pl.DataFrame] = []
    total_embedded = 0
    batch_start_time = time.time()

    for batch_idx in range(num_batches):
        start_row = batch_idx * batch_size
        end_row = min(start_row + batch_size, total_remaining)
        batch_df = remaining_df.slice(start_row, end_row - start_row)

        # Extract document texts
        documents = batch_df.select("document").to_series().to_list()

        try:
            # Generate embeddings
            embeddings = embedding_service.encode_batch(documents)

            # Build result rows
            result_df = _build_embedding_rows(batch_df, embeddings)
            accumulated_results.append(result_df)
            total_embedded += result_df.height

            # Progress logging
            elapsed = time.time() - batch_start_time
            docs_per_sec = total_embedded / elapsed if elapsed > 0 else 0
            progress_pct = (total_embedded / total_remaining) * 100
            eta_seconds = (total_remaining - total_embedded) / docs_per_sec if docs_per_sec > 0 else 0
            eta_min = eta_seconds / 60

            logger.info(
                f"Batch {batch_idx + 1}/{num_batches} | "
                f"{total_embedded:,}/{total_remaining:,} ({progress_pct:.1f}%) | "
                f"{docs_per_sec:.0f} docs/sec | "
                f"ETA: {eta_min:.1f} min"
            )

        except Exception as e:
            logger.error(f"Error encoding batch {batch_idx + 1}: {e}")
            logger.warning("Saving checkpoint before aborting...")
            # Save whatever we have so far
            if accumulated_results:
                partial = pl.concat(accumulated_results)
                if checkpoint_df is not None:
                    partial = pl.concat([checkpoint_df, partial])
                _save_checkpoint(partial)
            raise

        # Periodic checkpoint save
        if (batch_idx + 1) % checkpoint_interval == 0 and accumulated_results:
            merged = pl.concat(accumulated_results)
            if checkpoint_df is not None:
                merged = pl.concat([checkpoint_df, merged])
            _save_checkpoint(merged)
            logger.info(f"Checkpoint saved at batch {batch_idx + 1} ({merged.height:,} total embeddings).")

    # 6. Final consolidation
    logger.info("Consolidating all embeddings...")
    all_new = pl.concat(accumulated_results) if accumulated_results else pl.DataFrame()

    if checkpoint_df is not None and checkpoint_df.height > 0:
        final_df = pl.concat([checkpoint_df, all_new])
    else:
        final_df = all_new

    # Save final checkpoint
    _save_checkpoint(final_df)

    # 7. Export final output
    summary = _export_final(
        final_df, embeddings_dir, reports_dir, pipeline_start, resolved_device
    )
    summary["newly_embedded"] = total_embedded

    logger.info("=" * 60)
    logger.info("Embedding Generation Pipeline completed successfully!")
    logger.info("=" * 60)

    return summary


def _export_final(
    df: pl.DataFrame,
    embeddings_dir: Path,
    reports_dir: Path,
    pipeline_start: float,
    device: Optional[str],
) -> dict:
    """
    Exports the final embeddings Parquet and metadata JSON report.
    """
    total_time = time.time() - pipeline_start

    # Export final Parquet
    output_path = embeddings_dir / "movie_embeddings.parquet"
    df.write_parquet(output_path)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved final embeddings: {output_path} ({df.height:,} rows, {file_size_mb:.1f} MB)")

    # Determine embedding dimension from the data
    if df.height > 0:
        sample_emb = df.select("embedding").row(0)[0]
        embedding_dim = len(sample_emb) if sample_emb else 0
    else:
        embedding_dim = 0

    # Build metadata report
    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": settings.EMBEDDING_MODEL,
        "embedding_dimension": embedding_dim,
        "total_movies_embedded": df.height,
        "device": device or settings.DEVICE,
        "total_time_seconds": round(total_time, 2),
        "total_time_minutes": round(total_time / 60, 2),
        "output_file": str(output_path),
        "output_size_mb": round(file_size_mb, 2),
    }

    # Save metadata JSON alongside embeddings
    meta_path = embeddings_dir / "embedding_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved embedding metadata: {meta_path}")

    # Also save to reports directory
    report_path = reports_dir / "embedding_generation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved embedding report: {report_path}")

    logger.info(
        f"Pipeline summary: {df.height:,} movies embedded in {total_time / 60:.1f} min | "
        f"Dimension: {embedding_dim} | Device: {device or settings.DEVICE}"
    )

    return metadata


def main():
    """CLI entry point for the embedding generation pipeline."""
    parser = argparse.ArgumentParser(
        description="ChitraAI Embedding Generation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N movies (for testing)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override the default batch size."
    )
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda"], default=None,
        help="Force a specific compute device."
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore existing checkpoints and start from scratch."
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Do not attempt to resume from a checkpoint."
    )

    args = parser.parse_args()

    build_movie_embeddings(
        limit=args.limit,
        batch_size=args.batch_size,
        device=args.device,
        resume=not args.no_resume,
        fresh=args.fresh,
    )


if __name__ == "__main__":
    main()
