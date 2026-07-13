import math
from typing import Dict, Any, Union
from loguru import logger
import polars as pl

class DatasetProfiler:
    """
    Profiler for generating detailed structural and quality reports on Polars datasets.
    Operates in a completely read-only manner.
    """
    @staticmethod
    def profile_dataframe(name: str, frame: Union[pl.DataFrame, pl.LazyFrame]) -> Dict[str, Any]:
        """
        Profiles a Polars DataFrame or LazyFrame and returns a dictionary with the results.
        Uses optimized Polars expressions to run aggregation checks in a single pass.
        """
        logger.info(f"Profiling dataset component: '{name}'...")
        
        # Ensure we are working with a LazyFrame for optimized execution
        lf = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
        
        # 1. Compute dimensions
        row_count = lf.select(pl.len()).collect().item()
        schema = lf.collect_schema()
        column_names = schema.names()
        column_count = len(column_names)
        
        # 2. Schema Datatype Summary
        datatype_summary: Dict[str, int] = {}
        schema_dict = {}
        for col_name in column_names:
            dtype_str = str(schema.get(col_name))
            schema_dict[col_name] = dtype_str
            datatype_summary[dtype_str] = datatype_summary.get(dtype_str, 0) + 1

        missing_stats = {}
        unique_stats = {}

        if row_count > 0:
            # 3. Missing and Unique value count (Single pass using list of expressions)
            null_exprs = [pl.col(col).null_count().alias(f"{col}_null_count") for col in column_names]
            
            # Use approx_n_unique for very large datasets to prevent extreme memory use and CPU hangs
            if row_count > 5_000_000:
                unique_exprs = [pl.col(col).approx_n_unique().alias(f"{col}_unique_count") for col in column_names]
            else:
                unique_exprs = [pl.col(col).n_unique().alias(f"{col}_unique_count") for col in column_names]
            
            # Execute aggregations in a single collect call
            agg_results = lf.select(null_exprs + unique_exprs).collect().row(0)
            
            for idx, col in enumerate(column_names):
                null_cnt = agg_results[idx]
                uniq_cnt = agg_results[idx + column_count]
                
                null_pct = (null_cnt / row_count) * 100
                missing_stats[col] = {
                    "count": int(null_cnt),
                    "percentage": round(null_pct, 4)
                }
                unique_stats[col] = int(uniq_cnt)
        else:
            for col in column_names:
                missing_stats[col] = {"count": 0, "percentage": 0.0}
                unique_stats[col] = 0

        # 4. Duplicate statistics
        try:
            if row_count > 5_000_000:
                # Estimate duplicate count from the first 1,000,000 rows to prevent memory exhaustion
                sample_limit = 1_000_000
                sample_lf = lf.limit(sample_limit)
                sample_unique = sample_lf.unique().select(pl.len()).collect().item()
                sample_duplicates = sample_limit - sample_unique
                duplicate_count = int(sample_duplicates * (row_count / sample_limit))
                duplicate_percentage = (duplicate_count / row_count) * 100
                logger.info(f"[{name}] Estimated duplicates from 1M rows sample: {duplicate_count:,} ({duplicate_percentage:.4f}%)")
            else:
                unique_rows = lf.unique().select(pl.len()).collect().item()
                duplicate_count = row_count - unique_rows
                duplicate_percentage = (duplicate_count / row_count) * 100 if row_count > 0 else 0.0
        except Exception as e:
            logger.warning(f"Could not compute duplicate statistics for '{name}': {e}")
            duplicate_count = 0
            duplicate_percentage = 0.0


        # 5. Memory usage footprint estimation
        try:
            # Take a small sample to calculate average row footprint
            sample_df = lf.limit(100).collect()
            sample_size_bytes = sample_df.estimated_size()
            # Estimate total size
            estimated_memory_bytes = int((sample_size_bytes / 100) * row_count) if row_count > 0 else 0
        except Exception as e:
            logger.warning(f"Could not estimate memory usage for '{name}': {e}")
            estimated_memory_bytes = 0
            
        estimated_memory_mb = round(estimated_memory_bytes / (1024 * 1024), 4)

        # 6. Sample Records (first 5 rows)
        sample_records = []
        try:
            sample_df = lf.limit(5).collect()
            raw_records = sample_df.to_dicts()
            # Sanitize records for JSON compatibility (replace float NaN/Inf with None)
            for record in raw_records:
                sanitized = {}
                for key, val in record.items():
                    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                        sanitized[key] = None
                    else:
                        sanitized[key] = val
                sample_records.append(sanitized)
        except Exception as e:
            logger.warning(f"Could not fetch sample records for '{name}': {e}")

        logger.info(f"Profiling complete for '{name}': {row_count:,} rows, {column_count} columns.")

        return {
            "component_name": name,
            "row_count": row_count,
            "column_count": column_count,
            "schema": schema_dict,
            "datatype_summary": datatype_summary,
            "missing_values": missing_stats,
            "unique_values": unique_stats,
            "duplicate_statistics": {
                "duplicate_count": duplicate_count,
                "duplicate_percentage": round(duplicate_percentage, 4)
            },
            "estimated_memory_mb": estimated_memory_mb,
            "sample_records": sample_records
        }
