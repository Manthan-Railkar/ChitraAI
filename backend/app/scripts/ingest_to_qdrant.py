import os
import sys
import time
from pathlib import Path
from loguru import logger
import polars as pl
import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.services.embedding_service import EmbeddingService

def main():
    logger.info("Initializing Qdrant Cloud Ingestion Script...")
    
    # 1. Connect to Qdrant Cloud
    if not settings.QDRANT_URL:
        logger.error("QDRANT_URL is not set in backend settings. Please configure it in your .env file.")
        sys.exit(1)
        
    logger.info(f"Connecting to Qdrant Cloud at {settings.QDRANT_URL}...")
    try:
        qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
            timeout=30.0
        )
        qdrant_client.get_collections()
        logger.info("Successfully connected to Qdrant.")
    except Exception as e:
        logger.warning(f"Failed to connect to Qdrant at {settings.QDRANT_URL}: {e}")
        logger.warning(f"Falling back to local disk-based Qdrant client at {settings.QDRANT_PATH}...")
        try:
            qdrant_client = QdrantClient(path=settings.QDRANT_PATH)
            qdrant_client.get_collections()
            logger.info("Successfully initialized local disk-based Qdrant.")
        except Exception as local_e:
            logger.critical(f"Failed to initialize local disk-based Qdrant: {local_e}")
            sys.exit(1)

    # 2. Check metadata canonical file
    processed_dir = Path(settings.PROCESSED_DATA_DIR)
    tmdb_parquet_path = processed_dir / "canonical" / "tmdb_canonical.parquet"
    if not tmdb_parquet_path.exists():
        logger.error(f"tmdb_canonical.parquet not found at {tmdb_parquet_path}. Run rebuild_tmdb_canonical.py first.")
        sys.exit(1)
        
    logger.info(f"Loading TMDb canonical metadata from {tmdb_parquet_path}...")
    movies_df = pl.read_parquet(tmdb_parquet_path, columns=["tmdb_id"])
    logger.info(f"Loaded {movies_df.height:,} movies.")

    # 3. Check precomputed embeddings
    embeddings_dir = Path(settings.EMBEDDINGS_DIR)
    embeddings_path = embeddings_dir / "tmdb_embeddings.parquet"
    if not embeddings_path.exists():
        logger.error(f"tmdb_embeddings.parquet not found at {embeddings_path}. Run generate_tmdb_embeddings.py first.")
        sys.exit(1)
        
    logger.info(f"Loading precomputed embeddings from {embeddings_path}...")
    emb_df = pl.read_parquet(embeddings_path)
    
    # Join to keep only aligned movies
    aligned_df = movies_df.join(emb_df, on="tmdb_id", how="inner")
    logger.info(f"Aligned {aligned_df.height:,} movies with precomputed embeddings.")
    
    tmdb_ids = aligned_df.select("tmdb_id").to_series().to_numpy()
    raw_embs = np.vstack(list(aligned_df.select("embedding").to_series().to_numpy())).astype(np.float32)
    
    # L2-normalize embeddings (required for cosine similarity)
    norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    normalized_embs = raw_embs / norms
    
    # 4. Ensure collection exists
    embedding_service = EmbeddingService()
    vector_size = embedding_service.get_embedding_dimension()
    
    exists = False
    try:
        exists = qdrant_client.collection_exists(collection_name=settings.QDRANT_COLLECTION)
        if exists:
            info = qdrant_client.get_collection(collection_name=settings.QDRANT_COLLECTION)
            vectors_cfg = info.config.params.vectors
            existing_size = getattr(vectors_cfg, 'size', None)
            if existing_size is None and isinstance(vectors_cfg, dict):
                for v in vectors_cfg.values():
                    if hasattr(v, 'size'):
                        existing_size = v.size
                        break
            if existing_size is not None and existing_size != vector_size:
                logger.warning(f"Vector size mismatch: existing={existing_size}, expected={vector_size}. Recreating collection...")
                qdrant_client.delete_collection(collection_name=settings.QDRANT_COLLECTION)
                exists = False
    except Exception as e:
        logger.warning(f"Error checking collection details: {e}. Checking collection lists...")
        try:
            cols = qdrant_client.get_collections()
            exists = any(c.name == settings.QDRANT_COLLECTION for c in cols.collections)
            if exists:
                info = qdrant_client.get_collection(collection_name=settings.QDRANT_COLLECTION)
                vectors_cfg = info.config.params.vectors
                existing_size = getattr(vectors_cfg, 'size', None)
                if existing_size is not None and existing_size != vector_size:
                    qdrant_client.delete_collection(collection_name=settings.QDRANT_COLLECTION)
                    exists = False
        except Exception as ex:
            logger.error(f"Failed to check collections: {ex}. Assuming collection does not exist.")
            exists = False

    if not exists:
        logger.info(f"Collection '{settings.QDRANT_COLLECTION}' not found. Creating collection...")
        try:
            qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Collection '{settings.QDRANT_COLLECTION}' created successfully.")
        except Exception as e:
            logger.critical(f"Failed to create collection: {e}")
            sys.exit(1)
    else:
        logger.info(f"Collection '{settings.QDRANT_COLLECTION}' exists.")

    # 5. Ingest in batches
    batch_size = settings.QDRANT_BATCH_SIZE or 500
    total_points = len(tmdb_ids)
    uploaded_count = 0
    
    logger.info(f"Uploading {total_points} vectors to collection '{settings.QDRANT_COLLECTION}'...")
    
    for i in range(0, total_points, batch_size):
        batch_ids = tmdb_ids[i:i + batch_size]
        batch_vectors = normalized_embs[i:i + batch_size]
        
        # Check for existing points in this batch for resumability
        try:
            existing = qdrant_client.retrieve(
                collection_name=settings.QDRANT_COLLECTION,
                ids=[int(tid) for tid in batch_ids],
                with_payload=False,
                with_vectors=False
            )
            existing_ids = {p.id for p in existing}
        except Exception as e:
            logger.warning(f"Failed to check existing points for batch: {e}. Upserting all in batch.")
            existing_ids = set()
        
        points_to_upload = []
        for j, tid in enumerate(batch_ids):
            if int(tid) not in existing_ids:
                points_to_upload.append(
                    PointStruct(
                        id=int(tid),
                        vector=batch_vectors[j].tolist(),
                        payload={"tmdb_id": int(tid)}
                    )
                )
        
        if points_to_upload:
            for attempt in range(1, 4):
                try:
                    qdrant_client.upsert(
                        collection_name=settings.QDRANT_COLLECTION,
                        points=points_to_upload,
                        wait=True
                    )
                    uploaded_count += len(points_to_upload)
                    logger.info(f"Uploaded batch {i//batch_size + 1}: {len(points_to_upload)} vectors. Total uploaded: {uploaded_count}/{total_points}")
                    break
                except Exception as e:
                    if attempt == 3:
                        logger.error(f"Failed to upload batch after 3 attempts: {e}")
                        sys.exit(1)
                    logger.warning(f"Upsert failed (attempt {attempt}/3): {e}. Retrying in {1.0 * attempt}s...")
                    time.sleep(1.0 * attempt)
        else:
            logger.info(f"Batch {i//batch_size + 1} already exists. Skipping.")
            
    logger.info("Qdrant Cloud ingestion completed successfully!")

if __name__ == "__main__":
    main()
