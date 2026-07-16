import os
import sys
import argparse
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
import polars as pl
from tqdm.asyncio import tqdm

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.services.tmdb_service import TMDbService
from app.datasets.adapters import CANONICAL_SCHEMA, CANONICAL_COLUMNS


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for the enrichment script."""
    parser = argparse.ArgumentParser(description="TMDb Movie Metadata Enrichment Pipeline")
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of movies to enrich in this run (default: 1000). Set to 0 or negative for no limit."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for concurrent API requests (default: 50)."
    )
    return parser.parse_args()


def extract_enrichment_data(api_response: Dict[str, Any]) -> Dict[str, Any]:
    """Parses raw TMDb API response into canonical enrichment fields."""
    poster_path = api_response.get("poster_path")
    if poster_path and not poster_path.startswith("http"):
        poster_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
        
    backdrop_path = api_response.get("backdrop_path")
    if backdrop_path and not backdrop_path.startswith("http"):
        backdrop_path = f"https://image.tmdb.org/t/p/w1280{backdrop_path}"
    
    collection_name = None
    belongs = api_response.get("belongs_to_collection")
    if belongs and isinstance(belongs, dict):
        collection_name = belongs.get("name")
        
    trailer_url = None
    videos = api_response.get("videos", {}).get("results", [])
    for video in videos:
        if video.get("site") == "YouTube" and video.get("type") == "Trailer":
            trailer_url = f"https://www.youtube.com/watch?v={video.get('key')}"
            break
            
    streaming_providers = []
    providers = api_response.get("watch/providers", {}).get("results", {}).get("US", {}).get("flatrate", [])
    if providers:
        streaming_providers = [p.get("provider_name") for p in providers if p.get("provider_name")]
        
    certification = None
    releases = api_response.get("release_dates", {}).get("results", [])
    for r in releases:
        if r.get("iso_3166_1") == "US":
            dates = r.get("release_dates", [])
            for d in dates:
                cert = d.get("certification")
                if cert:
                    certification = cert
                    break
            if certification:
                break
    if not certification:
        for r in releases:
            dates = r.get("release_dates", [])
            for d in dates:
                cert = d.get("certification")
                if cert:
                    certification = cert
                    break
            if certification:
                break

    runtime = api_response.get("runtime")
    overview = api_response.get("overview")
    popularity = api_response.get("popularity")
    
    genres = []
    for g in api_response.get("genres", []):
        if g.get("name"):
            genres.append(g.get("name"))
            
    keywords = []
    kw_list = api_response.get("keywords", {}).get("keywords", [])
    if not kw_list:
        kw_list = api_response.get("keywords", {}).get("results", [])
    for kw in kw_list:
        if kw.get("name"):
            keywords.append(kw.get("name"))
            
    companies = []
    for c in api_response.get("production_companies", []):
        if c.get("name"):
            companies.append(c.get("name"))

    return {
        "poster_path": poster_path,
        "backdrop_path": backdrop_path,
        "trailer_url": trailer_url,
        "streaming_providers": streaming_providers if streaming_providers else None,
        "collection_name": collection_name,
        "certification": certification,
        "runtime_minutes": runtime,
        "overview": overview,
        "popularity": popularity,
        "genres": genres if genres else None,
        "keywords": keywords if keywords else None,
        "production_companies": companies if companies else None
    }


async def enrich_single_movie(
    tmdb_service: TMDbService,
    imdb_id: Optional[str],
    tmdb_id: Optional[int]
) -> Optional[Dict[str, Any]]:
    """Enriches a single movie by resolving IDs and fetching API details."""
    resolved_tmdb_id = tmdb_id
    
    # 1. Resolve TMDb ID via IMDb ID if missing
    if resolved_tmdb_id is None and imdb_id:
        try:
            resolved_tmdb_id = await tmdb_service.fetch_tmdb_id_by_imdb(imdb_id)
        except Exception as e:
            logger.debug(f"Failed to resolve TMDb ID for IMDb ID {imdb_id}: {e}")
            return None

    if resolved_tmdb_id is None:
        return None

    # 2. Fetch full details from TMDb API / local cache
    try:
        details = await tmdb_service.fetch_movie_details(resolved_tmdb_id)
        if details:
            parsed = extract_enrichment_data(details)
            parsed["imdb_id"] = imdb_id
            parsed["tmdb_id"] = resolved_tmdb_id
            return parsed
    except Exception as e:
        logger.debug(f"Failed to enrich movie TMDb ID {resolved_tmdb_id}: {e}")
        
    return None


async def run_enrichment_pipeline(limit: int, batch_size: int) -> None:
    """Orchestrates loading, processing, and saving enriched canonical knowledge base."""
    # Ensure logging is setup
    setup_logging()
    
    knowledge_base_path = Path(settings.PROCESSED_DATA_DIR) / "canonical" / "movies_knowledge_base.parquet"
    if not knowledge_base_path.exists():
        logger.error(f"Merged movies knowledge base not found at {knowledge_base_path}. Run preprocess first.")
        sys.exit(1)

    logger.info(f"Loading merged knowledge base from {knowledge_base_path}...")
    df = pl.read_parquet(knowledge_base_path)

    # If new columns are not present in the Parquet file, initialize them with nulls
    new_cols = {
        "poster_path": pl.String,
        "backdrop_path": pl.String,
        "trailer_url": pl.String,
        "streaming_providers": pl.List(pl.String),
        "collection_name": pl.String,
        "certification": pl.String
    }
    
    for col, dtype in new_cols.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))

    # Identify incomplete movie candidates
    # Candidates must have an imdb_id or tmdb_id, and miss at least one enriched field
    candidates_filter = (
        (pl.col("imdb_id").is_not_null() | pl.col("tmdb_id").is_not_null()) &
        (
            pl.col("poster_path").is_null() |
            pl.col("backdrop_path").is_null() |
            pl.col("trailer_url").is_null() |
            pl.col("streaming_providers").is_null() |
            pl.col("collection_name").is_null() |
            pl.col("certification").is_null()
        )
    )
    
    candidates_df = df.filter(candidates_filter)
    logger.info(f"Found {candidates_df.height:,} candidate movies requiring enrichment.")

    if candidates_df.height == 0:
        logger.info("All movie records are already enriched. Exiting.")
        return

    # Apply batch limit if specified
    if limit > 0:
        candidates_df = candidates_df.head(limit)
        logger.info(f"Limiting enrichment run to first {limit} movies.")

    # Select candidates keys
    keys = candidates_df.select(["imdb_id", "tmdb_id"]).to_dicts()

    tmdb_service = TMDbService()
    
    logger.info(f"Enriching records asynchronously (Concurrency cap: 10, Batch size: {batch_size})...")
    
    # Process tasks concurrently in batches
    enriched_results = []
    
    for i in range(0, len(keys), batch_size):
        batch = keys[i : i + batch_size]
        tasks = [
            enrich_single_movie(tmdb_service, item["imdb_id"], item["tmdb_id"])
            for item in batch
        ]
        
        # tqdm updates progress for each finished task in the batch
        batch_results = await tqdm.gather(*tasks, desc=f"Enriching Batch {i // batch_size + 1}")
        enriched_results.extend([r for r in batch_results if r is not None])

    if not enriched_results:
        logger.info("No movie records were successfully enriched in this run.")
        return

    logger.info(f"Successfully retrieved enrichment details for {len(enriched_results):,} movies.")

    # Convert results list to DataFrame
    # Let's map schema for results df to align with CANONICAL_SCHEMA
    results_schema = {col: CANONICAL_SCHEMA[col] for col in CANONICAL_COLUMNS if col in enriched_results[0]}
    # Include both imdb_id and tmdb_id as join keys
    results_schema["imdb_id"] = pl.String
    results_schema["tmdb_id"] = pl.Int64
    
    enriched_df = pl.DataFrame(enriched_results, schema=results_schema)

    # Perform updates by joining on tmdb_id (or imdb_id) and coalescing
    # We rename columns of enriched_df to avoid name collisions
    enrich_cols = [c for c in enriched_df.columns if c not in ["imdb_id", "tmdb_id"]]
    rename_mapping = {col: f"{col}_enriched" for col in enrich_cols}
    enriched_df_renamed = enriched_df.rename(rename_mapping)

    logger.info("Applying enriched values to movies knowledge base...")
    
    # We join on tmdb_id first, falling back to imdb_id for matching
    # To handle join keys safely, let's join on tmdb_id
    # (Since TMDb resolution resolved imdb_id to tmdb_id, tmdb_id is populated in enriched_df)
    df_joined = df.join(
        enriched_df_renamed.drop("imdb_id"),
        on="tmdb_id",
        how="left"
    )

    # For each enriched column, fill existing nulls with the enriched value
    update_exprs = []
    for col in enrich_cols:
        enriched_col_name = f"{col}_enriched"
        
        # If the column is a List type (genres, keywords, production_companies, streaming_providers),
        # we can either merge or fill_null.
        # "Existing valid values must never be overwritten." -> fill_null satisfies this perfectly!
        if isinstance(CANONICAL_SCHEMA[col], pl.List):
            expr = pl.col(col).fill_null(pl.col(enriched_col_name)).alias(col)
        else:
            expr = pl.col(col).fill_null(pl.col(enriched_col_name)).alias(col)
            
        update_exprs.append(expr)

    # Apply updates
    df_updated = df_joined.with_columns(update_exprs).drop([f"{col}_enriched" for col in enrich_cols])

    # Regenerate natural language movie documents for embedding
    from app.datasets.document_generator import generate_knowledge_base_documents
    logger.info("Regenerating natural-language movie documents for embedding...")
    df_updated = generate_knowledge_base_documents(df_updated)

    # Reorder columns to CANONICAL_COLUMNS layout
    df_final = df_updated.select(CANONICAL_COLUMNS)

    logger.info(f"Saving enriched canonical knowledge base back to {knowledge_base_path}...")
    df_final.write_parquet(knowledge_base_path)

    
    logger.info("Metadata enrichment pipeline completed successfully.")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_enrichment_pipeline(limit=args.limit, batch_size=args.batch_size))
