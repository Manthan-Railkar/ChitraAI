import os
import sys
import time
from pathlib import Path
from loguru import logger
import polars as pl
import numpy as np

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.core.model_manager import ModelManager
from app.services.embedding_service import EmbeddingService
from app.services.local_retrieval import build_embedding_document

def main():
    logger.info("Initializing embedding model...")
    ModelManager.load_model()
    
    embedding_service = EmbeddingService()
    processed_dir = Path(settings.PROCESSED_DATA_DIR)
    canonical_dir = processed_dir / "canonical"
    embeddings_dir = Path(settings.EMBEDDINGS_DIR)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    tmdb_parquet_path = canonical_dir / "tmdb_canonical.parquet"
    if not tmdb_parquet_path.exists():
        logger.error(f"tmdb_canonical.parquet not found at {tmdb_parquet_path}. Run rebuild_tmdb_canonical.py first.")
        sys.exit(1)
        
    logger.info(f"Loading TMDb canonical metadata from {tmdb_parquet_path}...")
    movies_df = pl.read_parquet(tmdb_parquet_path)
    logger.info(f"Loaded {movies_df.height:,} movies.")
    
    logger.info("Building metadata documents to embed...")
    docs = []
    tmdb_ids = movies_df.select("tmdb_id").to_series().to_list()
    titles = movies_df.select("title").to_series().to_list()
    taglines = movies_df.select("tagline").to_series().to_list()
    overviews = movies_df.select("overview").to_series().to_list()
    genres_list = movies_df.select("genres").to_series().to_list()
    keywords_list = movies_df.select("keywords").to_series().to_list()
    cast_list = movies_df.select("cast").to_series().to_list()
    directors_list = movies_df.select("directors").to_series().to_list()
    
    for idx in range(len(tmdb_ids)):
        doc = build_embedding_document(
            title=titles[idx],
            tagline=taglines[idx],
            overview=overviews[idx],
            genres=genres_list[idx],
            keywords=keywords_list[idx],
            cast=cast_list[idx],
            directors=directors_list[idx]
        )
        docs.append(doc)
        
    output_path = embeddings_dir / "tmdb_embeddings.parquet"
    logger.info(f"Generating embeddings for {len(docs):,} movies using {settings.EMBEDDING_MODEL}...")
    
    batch_size = 256
    embeddings_all = []
    
    start_time = time.time()
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i : i + batch_size]
        batch_embs = embedding_service.encode_batch(batch_docs, normalize=True)
        embeddings_all.append(batch_embs)
        
        elapsed = time.time() - start_time
        processed = min(i + batch_size, len(docs))
        speed = processed / elapsed if elapsed > 0 else 0
        eta = (len(docs) - processed) / speed if speed > 0 else 0
        logger.info(f"Embedded {processed:,}/{len(docs):,} movies | speed={speed:.1f} docs/s | elapsed={elapsed:.1f}s | ETA={eta/60:.1f} min")
        
    embeddings_matrix = np.vstack(embeddings_all)
    
    logger.info(f"Saving embeddings parquet to {output_path}...")
    emb_lists = [emb.tolist() for emb in embeddings_matrix]
    emb_df = pl.DataFrame({
        "tmdb_id": tmdb_ids,
        "embedding": emb_lists
    })
    emb_df.write_parquet(output_path)
    logger.info("Local movie embeddings generation completed successfully!")

if __name__ == "__main__":
    main()
