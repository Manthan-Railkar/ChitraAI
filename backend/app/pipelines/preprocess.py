import os
import sys
import json
from pathlib import Path
from loguru import logger
import polars as pl

# Add the backend directory to sys.path to allow imports from app
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.datasets.loader import DatasetIngestionSystem
from app.datasets.profiler import DatasetProfiler
from app.datasets.validator import DatasetValidator
from app.datasets.cleaner import DatasetCleaningPipeline
from app.datasets.adapters import IMDbAdapter, TMDbAdapter, MovieLensAdapter, WikipediaAdapter
from app.datasets.merger import DatasetMerger
from app.pipelines.export_qa import run_qa_and_export


def preprocess_datasets() -> None:
    """
    Runs the full ingestion, validation, profiling, cleaning, and canonical mapping pipeline.
    """
    # Setup centralized logging
    setup_logging()
    
    logger.info("Initializing ChitraAI Preprocessing Pipeline...")
    
    # 1. Dataset Discovery & Ingestion Validation
    ingestion_system = DatasetIngestionSystem()
    discovered = ingestion_system.discover_datasets()
    if not discovered:
        logger.error("No dataset directories discovered in the raw folder. Pipeline aborted.")
        sys.exit(1)
        
    validation_results = ingestion_system.validate_all()
    if not all(validation_results.values()):
        logger.warning(f"File validation warnings detected: {validation_results}")
        
    logger.info("Loading all raw datasets lazily into memory...")
    dataset_frames = ingestion_system.load_all(lazy=True)
    
    # Ensure reports directory exists
    reports_path = Path(settings.REPORTS_DIR)
    reports_path.mkdir(parents=True, exist_ok=True)
    
    # 2. Dataset Profiling
    logger.info("Running dataset profiling checks...")
    profiling_reports = {}
    
    for ds_name, files_dict in dataset_frames.items():
        profiling_reports[ds_name] = {}
        for file_name, frame in files_dict.items():
            try:
                # Profile components
                profile = DatasetProfiler.profile_dataframe(f"{ds_name}/{file_name}", frame)
                profiling_reports[ds_name][file_name] = profile
            except Exception as e:
                logger.error(f"Failed to profile '{ds_name}/{file_name}': {e}")
                
        # Save dataset-level profiling report
        ds_report_file = reports_path / f"{ds_name}_profiling_report.json"
        try:
            with open(ds_report_file, "w", encoding="utf-8") as f:
                json.dump(profiling_reports[ds_name], f, indent=2, ensure_ascii=False)
            logger.info(f"Saved profiling report to {ds_report_file.name}")
        except Exception as e:
            logger.error(f"Failed to write profiling report for dataset '{ds_name}': {e}")

    # 3. Dataset Validation
    logger.info("Running dataset validation and data quality checks...")
    try:
        validator = DatasetValidator(dataset_frames)
        validation_report = validator.run_all_validations()
        
        # Save validation report
        val_report_file = reports_path / "validation_report.json"
        with open(val_report_file, "w", encoding="utf-8") as f:
            json.dump(validation_report, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved validation report to {val_report_file.name}")
        
        # Log validation issues summary
        for ds_key, report in validation_report.items():
            status = report.get("validation_status", "UNKNOWN")
            issues_cnt = len(report.get("issues", []))
            
            if issues_cnt > 0:
                logger.warning(
                    f"[{report.get('dataset', ds_key)}] Validation Status: {status} | "
                    f"Found {issues_cnt} data quality issues."
                )
            else:
                logger.info(f"[{report.get('dataset', ds_key)}] Validation Status: {status} | No issues found.")
    except Exception as e:
        logger.error(f"Validation engine execution failed: {e}")

    # 4. Dataset Cleaning and Normalization
    logger.info("Running dataset cleaning and normalization pipeline...")
    try:
        cleaning_pipeline = DatasetCleaningPipeline(dataset_frames)
        cleaning_summary = cleaning_pipeline.run_cleaning()
        
        # Save cleaning summary report
        clean_summary_file = reports_path / "cleaning_summary.json"
        with open(clean_summary_file, "w", encoding="utf-8") as f:
            json.dump(cleaning_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved cleaning summary report to {clean_summary_file.name}")
        
        logger.info("Dataset cleaning completed successfully.")
    except Exception as e:
        logger.error(f"Cleaning and normalization pipeline failed: {e}")
        sys.exit(1)

    # 5. Canonical Schema Adaptation & Mapping
    logger.info("Running canonical schema mapping and adaptation...")
    processed_dir = Path(settings.PROCESSED_DATA_DIR)
    canonical_dir = processed_dir / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load cleaned Parquet files lazily to build adapter inputs
        logger.debug("Scanning cleaned files lazily...")
        
        # IMDb
        imdb_clean = {
            "title.basics.tsv.gz": pl.scan_parquet(processed_dir / "imdb_title_basics.parquet"),
            "title.ratings.tsv.gz": pl.scan_parquet(processed_dir / "imdb_title_ratings.parquet"),
            "title.principals.tsv.gz": pl.scan_parquet(processed_dir / "imdb_title_principals.parquet"),
            "title.crew.tsv.gz": pl.scan_parquet(processed_dir / "imdb_title_crew.parquet"),
            "name.basics.tsv.gz": pl.scan_parquet(processed_dir / "imdb_name_basics.parquet")
        }
        
        # MovieLens
        movielens_clean = {
            "movies.csv": pl.scan_parquet(processed_dir / "movielens_movies.parquet"),
            "ratings.csv": pl.scan_parquet(processed_dir / "movielens_ratings.parquet"),
            "tags.csv": pl.scan_parquet(processed_dir / "movielens_tags.parquet"),
            "links.csv": pl.scan_parquet(processed_dir / "movielens_links.parquet")
        }

        # TMDb
        tmdb_clean = {
            "movies_metadata.csv": pl.scan_parquet(processed_dir / "tmdb_movies_metadata.parquet"),
            "credits.csv": pl.scan_parquet(processed_dir / "tmdb_credits.parquet"),
            "keywords.csv": pl.scan_parquet(processed_dir / "tmdb_keywords.parquet")
        }

        # Wikipedia
        wikipedia_clean = {
            "wiki_movie_plots_deduped.csv": pl.scan_parquet(processed_dir / "wikipedia_movie_plots.parquet")
        }

        # Run IMDb adaptation
        logger.info("[Adapters] Running IMDb canonical mapping...")
        imdb_canonical = IMDbAdapter().adapt(imdb_clean).collect()
        imdb_canonical.write_parquet(canonical_dir / "imdb_canonical.parquet")
        logger.info(f"[Adapters] IMDb canonical table saved ({imdb_canonical.height:,} rows).")

        # Run TMDb adaptation
        logger.info("[Adapters] Running TMDb canonical mapping...")
        tmdb_canonical = TMDbAdapter().adapt(tmdb_clean).collect()
        tmdb_canonical.write_parquet(canonical_dir / "tmdb_canonical.parquet")
        logger.info(f"[Adapters] TMDb canonical table saved ({tmdb_canonical.height:,} rows).")

        # Run MovieLens adaptation
        logger.info("[Adapters] Running MovieLens canonical mapping...")
        movielens_canonical = MovieLensAdapter().adapt(movielens_clean).collect()
        movielens_canonical.write_parquet(canonical_dir / "movielens_canonical.parquet")
        logger.info(f"[Adapters] MovieLens canonical table saved ({movielens_canonical.height:,} rows).")

        # Run Wikipedia adaptation
        logger.info("[Adapters] Running Wikipedia canonical mapping...")
        wikipedia_canonical = WikipediaAdapter().adapt(wikipedia_clean).collect()
        wikipedia_canonical.write_parquet(canonical_dir / "wikipedia_canonical.parquet")
        logger.info(f"[Adapters] Wikipedia canonical table saved ({wikipedia_canonical.height:,} rows).")

        logger.info("All datasets successfully mapped and saved to canonical format.")
    except Exception as e:
        logger.error(f"Canonical schema mapping failed: {e}")
        sys.exit(1)

    # 6. Unified Knowledge Base Merging
    logger.info("Running dataset merging and knowledge base consolidation...")
    try:
        merger = DatasetMerger()
        merge_summary = merger.run_merging()
        
        # Save merge summary report
        merge_summary_file = reports_path / "merge_summary.json"
        with open(merge_summary_file, "w", encoding="utf-8") as f:
            json.dump(merge_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved merge summary report to {merge_summary_file.name}")
        
        logger.info("Dataset merging completed successfully.")
    except Exception as e:
        logger.error(f"Dataset merging failed: {e}")
        sys.exit(1)

    # 7. Quality Assurance & Final Export
    logger.info("Running final quality assurance and multi-format export...")
    try:
        qa_report = run_qa_and_export()

        # Save QA summary to reports
        qa_summary_file = reports_path / "qa_export_summary.json"
        with open(qa_summary_file, "w", encoding="utf-8") as f:
            json.dump(qa_report, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved QA export summary report to {qa_summary_file.name}")

        final_stats = qa_report.get("dataset_statistics", {})
        logger.info(
            f"QA complete: {final_stats.get('initial_rows', '?')} rows validated, "
            f"{final_stats.get('removed_rows', '?')} rows removed, "
            f"{final_stats.get('final_rows', '?')} rows exported."
        )
    except Exception as e:
        logger.error(f"Quality assurance and export pipeline failed: {e}")
        sys.exit(1)

    logger.info(
        "Pipeline execution completed successfully. "
        "Ingestion, Validation, Profiling, Normalization, Canonical Mapping, "
        "Merging, QA & Export complete."
    )


if __name__ == "__main__":
    preprocess_datasets()
