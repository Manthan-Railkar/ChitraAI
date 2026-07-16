import os
import sys
from pathlib import Path
from loguru import logger
import polars as pl

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.datasets.adapters import TMDbAdapter, CANONICAL_COLUMNS, CANONICAL_SCHEMA
from app.datasets.merger import DatasetMerger

def align_parquet_schema(file_path: Path):
    """Loads a parquet file, ensures it has all CANONICAL_COLUMNS, and saves it back."""
    if not file_path.exists():
        logger.warning(f"File {file_path} does not exist. Skipping alignment.")
        return
    
    logger.info(f"Aligning schema of {file_path.name}...")
    df = pl.read_parquet(file_path)
    
    # Check if all canonical columns are present
    missing_cols = [col for col in CANONICAL_COLUMNS if col not in df.columns]
    if missing_cols:
        logger.info(f"Adding missing canonical columns {missing_cols} to {file_path.name}")
        exprs = []
        for col in CANONICAL_COLUMNS:
            if col in df.columns:
                exprs.append(pl.col(col).cast(CANONICAL_SCHEMA[col]))
            else:
                exprs.append(pl.lit(None).cast(CANONICAL_SCHEMA[col]).alias(col))
        df = df.select(exprs)
        df.write_parquet(file_path)
        logger.info(f"Successfully aligned and saved {file_path.name}")
    else:
        logger.info(f"{file_path.name} is already aligned.")

def main():
    logger.info("Starting TMDb canonical and merged knowledge base rebuild...")
    
    processed_dir = Path(settings.PROCESSED_DATA_DIR)
    canonical_dir = processed_dir / "canonical"
    
    # 1. Load processed TMDb parquets
    logger.info("Loading processed TMDb datasets...")
    tmdb_clean = {
        "movies_metadata.csv": pl.scan_parquet(processed_dir / "tmdb_movies_metadata.parquet"),
        "credits.csv": pl.scan_parquet(processed_dir / "tmdb_credits.parquet"),
        "keywords.csv": pl.scan_parquet(processed_dir / "tmdb_keywords.parquet")
    }
    
    # 2. Run TMDb adaptation
    logger.info("Running TMDb Adapter with tagline mapping...")
    tmdb_canonical = TMDbAdapter().adapt(tmdb_clean).collect()
    
    # 3. Save new TMDb canonical
    tmdb_canonical_path = canonical_dir / "tmdb_canonical.parquet"
    tmdb_canonical.write_parquet(tmdb_canonical_path)
    logger.info(f"Successfully wrote {tmdb_canonical.height:,} rows to {tmdb_canonical_path}")
    
    # 4. Align other canonical parquets on disk to have 'tagline' column
    align_parquet_schema(canonical_dir / "imdb_canonical.parquet")
    align_parquet_schema(canonical_dir / "movielens_canonical.parquet")
    align_parquet_schema(canonical_dir / "wikipedia_canonical.parquet")
    
    # 5. Rerun Merger to update the unified movies_knowledge_base.parquet
    logger.info("Running Dataset Merger to update movies_knowledge_base.parquet...")
    merger = DatasetMerger()
    summary = merger.run_merging()
    
    logger.info(f"Rebuild completed successfully. Merge Summary: {summary}")

if __name__ == "__main__":
    main()
