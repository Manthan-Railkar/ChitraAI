import os
import time
import gzip
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from loguru import logger
import polars as pl
from app.core.config import settings

def detect_compression(file_path: Path) -> Optional[str]:
    """
    Detects compression type based on file extension.
    Currently supports gzip (.gz).
    """
    if file_path.suffix == ".gz":
        return "gzip"
    return None

def detect_encoding_and_delimiter(file_path: Path, compression: Optional[str] = None) -> Tuple[str, str]:
    """
    Automatically detects encoding and column delimiter by sniffing a sample of the file.
    Tries utf-8, latin-1, and fallbacks. Sniffs commas, tabs, semicolons, and pipes.
    """
    encoding = "utf-8"
    delimiter = ","

    # Set initial default delimiter by suffix
    if ".tsv" in file_path.suffixes:
        delimiter = "\t"

    try:
        if compression == "gzip":
            with gzip.open(file_path, "rb") as f:
                sample_bytes = f.read(15000)
        else:
            with open(file_path, "rb") as f:
                sample_bytes = f.read(15000)
    except Exception as e:
        logger.warning(f"Error reading sample bytes from {file_path.name}: {e}. Using default encoding and delimiter.")
        return encoding, delimiter

    if not sample_bytes:
        return encoding, delimiter

    # Detect encoding
    detected_encoding = None
    for enc in ["utf-8", "latin-1", "utf-16", "cp1252"]:
        try:
            sample_bytes.decode(enc)
            detected_encoding = enc
            break
        except UnicodeDecodeError:
            continue

    if detected_encoding:
        encoding = detected_encoding
    else:
        encoding = "latin-1"  # Safe fallback for binary/mixed text

    # Sniff delimiter from decoded lines
    try:
        decoded_str = sample_bytes.decode(encoding, errors="ignore")
        lines = [line.strip() for line in decoded_str.split("\n") if line.strip()]
        if lines:
            header = lines[0]
            candidates = [",", "\t", ";", "|"]
            counts = {c: header.count(c) for c in candidates}
            best_candidate = max(counts, key=counts.get)
            if counts[best_candidate] > 0:
                delimiter = best_candidate
    except Exception as e:
        logger.warning(f"Error sniffing delimiter for {file_path.name}: {e}. Defaulting to '{delimiter}'.")

    return encoding, delimiter

def load_file_to_polars(
    file_path: Path,
    delimiter: str,
    encoding: str,
    compression: Optional[str],
    lazy: bool = False
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """
    Loads a text dataset file into a Polars DataFrame or LazyFrame.
    Attempts lazy Polars scanning first (if uncompressed), falls back to eager Polars reading,
    and finally falls back to Pandas read_csv with Conversion to Polars.
    """
    logger.info(f"Loading {file_path.name} (delimiter={repr(delimiter)}, encoding={encoding}, compression={compression}, lazy={lazy})")
    
    can_lazy = lazy and (compression is None)
    
    # Determine quote character configuration
    # TSVs (like IMDb) should not parse quotes as field qualifiers, as double quotes often appear literally.
    import csv
    if delimiter == "\t":
        quote_char = None
        pandas_quoting = csv.QUOTE_NONE
    else:
        quote_char = '"'
        pandas_quoting = csv.QUOTE_MINIMAL

    # 1. Attempt Polars Lazy Load (if applicable)
    if can_lazy:
        try:
            # Polars scan_csv only supports UTF-8 (and lossy).
            polars_encoding = "utf8-lossy" if encoding.lower() in ("utf-8", "utf8") else "utf8"
            lf = pl.scan_csv(
                str(file_path),
                separator=delimiter,
                quote_char=quote_char,
                encoding=polars_encoding,
                null_values=["\\N", "NA", "NaN", ""],
                ignore_errors=True,
                infer_schema_length=10000
            )
            # Perform a quick dry run to verify parsing success
            lf.limit(5).collect()
            logger.debug(f"Successfully loaded {file_path.name} lazily using Polars.")
            return lf
        except Exception as e:
            logger.warning(f"Polars lazy scan failed for {file_path.name}: {e}. Falling back to eager load.")

    # 2. Attempt Polars Eager Load
    try:
        polars_encoding = "utf8-lossy" if encoding.lower() in ("utf-8", "utf8") else "utf8"
        # pl.read_csv handles gzip natively
        df = pl.read_csv(
            str(file_path),
            separator=delimiter,
            quote_char=quote_char,
            encoding=polars_encoding,
            null_values=["\\N", "NA", "NaN", ""],
            ignore_errors=True,
            infer_schema_length=10000
        )
        logger.debug(f"Successfully loaded {file_path.name} eagerly using Polars.")
        return df.lazy() if lazy else df
    except Exception as e:
        logger.warning(f"Polars eager read failed for {file_path.name}: {e}. Falling back to Pandas.")

    # 3. Attempt Pandas Eager Load as Fallback
    try:
        pd_df = pd.read_csv(
            str(file_path),
            sep=delimiter,
            quoting=pandas_quoting,
            encoding=encoding,
            compression="gzip" if compression == "gzip" else None,
            keep_default_na=True,
            na_values=["\\N", "NA", "NaN"],
            low_memory=False,
            on_bad_lines="skip"
        )
        df = pl.from_pandas(pd_df)
        logger.debug(f"Successfully loaded {file_path.name} using Pandas fallback.")
        return df.lazy() if lazy else df
    except Exception as e:
        logger.error(f"All loaders failed for {file_path.name}: {e}")
        raise RuntimeError(f"Failed to load file {file_path}: {e}") from e

class BaseDatasetLoader:
    """
    Abstract base class defining a unified interface for all dataset loaders.
    """
    def __init__(self, name: str, raw_dir: Path, required_files: List[str]) -> None:
        self.name = name
        self.raw_dir = raw_dir
        self.required_files = required_files
        self.frames: Dict[str, Union[pl.DataFrame, pl.LazyFrame]] = {}
        self.loading_stats: Dict[str, Dict[str, Any]] = {}

    def validate(self) -> bool:
        """
        Validates that all required files exist in the dataset directory.
        """
        missing_files = []
        for file_name in self.required_files:
            file_path = self.raw_dir / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            logger.error(f"[{self.name}] Validation failed. Missing files: {missing_files}")
            return False
        
        logger.info(f"[{self.name}] Validation successful. All {len(self.required_files)} files found.")
        return True

    def load(self, lazy: bool = True) -> Dict[str, Union[pl.DataFrame, pl.LazyFrame]]:
        """
        Loads all required files and stores them in the frames dictionary.
        Tracks progress, performance timing, and logs metadata.
        """
        if not self.validate():
            raise FileNotFoundError(f"Missing required files for {self.name} dataset in {self.raw_dir}")

        self.frames = {}
        self.loading_stats = {}
        total_files = len(self.required_files)

        logger.info(f"[{self.name}] Starting dataset loading (lazy={lazy})...")
        for i, file_name in enumerate(self.required_files, 1):
            file_path = self.raw_dir / file_name
            start_time = time.perf_counter()

            # Detect format characteristics
            compression = detect_compression(file_path)
            encoding, delimiter = detect_encoding_and_delimiter(file_path, compression)

            # Load the file
            try:
                frame = load_file_to_polars(
                    file_path=file_path,
                    delimiter=delimiter,
                    encoding=encoding,
                    compression=compression,
                    lazy=lazy
                )
                self.frames[file_name] = frame
                elapsed = time.perf_counter() - start_time
                
                # Fetch basic dimensions for progress log
                is_lazy_frame = isinstance(frame, pl.LazyFrame)
                if is_lazy_frame:
                    row_count = frame.select(pl.len()).collect().item()
                    cols = frame.collect_schema().names()
                else:
                    row_count = frame.height
                    cols = frame.columns


                logger.info(
                    f"[{self.name}] Progress: [{i}/{total_files}] Loaded {file_name} "
                    f"({row_count:,} rows, {len(cols)} columns) in {elapsed:.2f}s"
                )

                # Store loading metadata for statistics reporting
                self.loading_stats[file_name] = {
                    "path": str(file_path),
                    "disk_size_bytes": file_path.stat().st_size,
                    "row_count": row_count,
                    "columns": cols,
                    "delimiter": delimiter,
                    "encoding": encoding,
                    "compression": compression or "None",
                    "elapsed_seconds": elapsed,
                    "is_lazy": is_lazy_frame
                }

            except Exception as e:
                logger.error(f"[{self.name}] Failed to load file {file_name}: {e}")
                raise

        logger.info(f"[{self.name}] Finished loading dataset successfully.")
        return self.frames

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Compiles and returns summary statistics of all loaded files.
        Includes file properties, row counts, memory footprints, and schemas.
        """
        stats = {}
        for file_name, frame in self.frames.items():
            load_stat = self.loading_stats.get(file_name, {})
            is_lazy = load_stat.get("is_lazy", True)

            # Retrieve schema
            schema = {}
            if is_lazy:
                schema = {name: str(dtype) for name, dtype in frame.collect_schema().items()}
                # Smart memory footprint estimation for LazyFrame
                row_count = load_stat.get("row_count", 0)
                try:
                    sample_df = frame.limit(100).collect()
                    sample_size = sample_df.estimated_size()
                    estimated_memory = int((sample_size / 100) * row_count) if row_count > 0 else 0
                except Exception as e:
                    logger.warning(f"Could not estimate memory size for lazy frame {file_name}: {e}")
                    estimated_memory = 0
            else:
                schema = {name: str(dtype) for name, dtype in frame.schema.items()}
                estimated_memory = frame.estimated_size()
                row_count = frame.height


            stats[file_name] = {
                "file_name": file_name,
                "disk_size_mb": round(load_stat.get("disk_size_bytes", 0) / (1024 * 1024), 2),
                "estimated_memory_mb": round(estimated_memory / (1024 * 1024), 2),
                "row_count": row_count,
                "column_count": len(load_stat.get("columns", [])),
                "columns": load_stat.get("columns", []),
                "schema": schema,
                "delimiter": repr(load_stat.get("delimiter", ",")),
                "encoding": load_stat.get("encoding", "utf-8"),
                "compression": load_stat.get("compression", "None"),
                "is_lazy": is_lazy
            }
        return stats


class IMDbLoader(BaseDatasetLoader):
    """
    Dataset loader specific to IMDb (.tsv.gz) files.
    """
    def __init__(self, raw_dir: Path) -> None:
        required_files = [
            "title.basics.tsv.gz",
            "title.ratings.tsv.gz",
            "title.principals.tsv.gz",
            "title.crew.tsv.gz",
            "name.basics.tsv.gz"
        ]
        super().__init__("IMDb", raw_dir, required_files)


class TMDbLoader(BaseDatasetLoader):
    """
    Dataset loader specific to Kaggle TMDb (.csv) files.
    """
    def __init__(self, raw_dir: Path) -> None:
        required_files = [
            "movies_metadata.csv",
            "credits.csv",
            "keywords.csv",
            "links.csv",
            "links_small.csv",
            "ratings_small.csv"
        ]
        super().__init__("TMDb", raw_dir, required_files)


class MovieLensLoader(BaseDatasetLoader):
    """
    Dataset loader specific to MovieLens (.csv) files.
    """
    def __init__(self, raw_dir: Path) -> None:
        required_files = [
            "movies.csv",
            "ratings.csv",
            "tags.csv",
            "links.csv"
        ]
        super().__init__("MovieLens", raw_dir, required_files)


class WikipediaLoader(BaseDatasetLoader):
    """
    Dataset loader specific to Wikipedia (.csv) files.
    """
    def __init__(self, raw_dir: Path) -> None:
        required_files = [
            "wiki_movie_plots_deduped.csv"
        ]
        super().__init__("Wikipedia", raw_dir, required_files)


class DatasetIngestionSystem:
    """
    Unified manager class for discovery, validation, and ingestion of all raw datasets.
    """
    def __init__(self, raw_data_dir: Optional[str] = None) -> None:
        self.raw_data_dir = Path(raw_data_dir or settings.RAW_DATA_DIR)
        self.loaders: Dict[str, BaseDatasetLoader] = {}
        logger.info(f"DatasetIngestionSystem initialized with raw directory: {self.raw_data_dir}")

    def discover_datasets(self) -> List[str]:
        """
        Scans the raw data directory for known dataset folders and initializes their loaders.
        """
        logger.info(f"Scanning '{self.raw_data_dir}' for dataset folders...")
        discovered = []

        # Supported dataset directories
        dataset_mappings = {
            "imdb": IMDbLoader,
            "tmdb": TMDbLoader,
            "movielens": MovieLensLoader,
            "wikipedia": WikipediaLoader
        }

        if not self.raw_data_dir.exists():
            logger.error(f"Raw data directory does not exist: {self.raw_data_dir}")
            return discovered

        for child in self.raw_data_dir.iterdir():
            if child.is_dir():
                name_lower = child.name.lower()
                if name_lower in dataset_mappings:
                    loader_cls = dataset_mappings[name_lower]
                    self.loaders[name_lower] = loader_cls(child)
                    discovered.append(child.name)
                    logger.info(f"Discovered dataset folder: '{child.name}'. Mapping to {loader_cls.__name__}.")

        logger.info(f"Discovery complete. Found {len(discovered)} dataset loaders.")
        return discovered

    def validate_all(self) -> Dict[str, bool]:
        """
        Validates all discovered datasets to ensure they contain all required files.
        """
        logger.info("Validating all discovered datasets...")
        validation_results = {}
        for name, loader in self.loaders.items():
            validation_results[name] = loader.validate()
        return validation_results

    def load_dataset(self, name: str, lazy: bool = True) -> Dict[str, Union[pl.DataFrame, pl.LazyFrame]]:
        """
        Loads a single dataset by name.
        """
        name_lower = name.lower()
        if name_lower not in self.loaders:
            raise ValueError(f"Dataset '{name}' is not discovered or supported.")
        return self.loaders[name_lower].load(lazy=lazy)

    def load_all(self, lazy: bool = True) -> Dict[str, Dict[str, Union[pl.DataFrame, pl.LazyFrame]]]:
        """
        Loads all discovered and valid datasets.
        """
        logger.info("Starting ingestion of all datasets...")
        all_data = {}
        for name, loader in self.loaders.items():
            try:
                all_data[name] = loader.load(lazy=lazy)
            except Exception as e:
                logger.error(f"Failed to load dataset '{name}': {e}. Skipping.")
        return all_data

    def report_summary_statistics(self) -> Dict[str, Any]:
        """
        Aggregates and logs summary statistics for all loaded datasets.
        """
        summary = {}
        logger.info("================================================================================")
        logger.info("                     DATASET INGESTION SUMMARY STATISTICS                       ")
        logger.info("================================================================================")
        
        for name, loader in self.loaders.items():
            if not loader.frames:
                logger.warning(f"Dataset '{name}' has not been loaded yet. No stats available.")
                continue
            
            loader_stats = loader.get_summary_statistics()
            summary[name] = loader_stats
            
            logger.info(f"Dataset: {loader.name}")
            for file_name, stats in loader_stats.items():
                logger.info(
                    f"  - {file_name}:"
                )
                logger.info(
                    f"      * Rows: {stats['row_count']:,} | Cols: {stats['column_count']} | IsLazy: {stats['is_lazy']}"
                )
                logger.info(
                    f"      * Disk: {stats['disk_size_mb']} MB | Est Memory: {stats['estimated_memory_mb']} MB"
                )
                logger.info(
                    f"      * Delimiter: {stats['delimiter']} | Encoding: {stats['encoding']} | Compression: {stats['compression']}"
                )
        logger.info("================================================================================")
        return summary
