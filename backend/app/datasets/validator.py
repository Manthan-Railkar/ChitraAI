import re
from typing import Dict, List, Any, Optional, Union
from loguru import logger
import polars as pl

class DatasetValidator:
    """
    Validation engine that executes extensive data quality, structural, and integrity
    checks on movie datasets without performing any cleaning or modification.
    """
    def __init__(self, dataset_frames: Dict[str, Dict[str, Union[pl.DataFrame, pl.LazyFrame]]]) -> None:
        """
        Initializes the validator with loaded dataset frames.
        Frames are stored as LazyFrames to allow optimized query planning.
        """
        self.datasets: Dict[str, Dict[str, pl.LazyFrame]] = {}
        for ds_name, files_dict in dataset_frames.items():
            self.datasets[ds_name] = {}
            for file_name, frame in files_dict.items():
                self.datasets[ds_name][file_name] = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
        
        self.reports: Dict[str, Any] = {}
        logger.info("DatasetValidator initialized.")

    def run_all_validations(self) -> Dict[str, Any]:
        """
        Runs validations for all datasets and cross-dataset relationships.
        Returns a compiled validation report.
        """
        logger.info("Executing dataset validation suite...")
        
        self.reports = {
            "imdb": self.validate_imdb(),
            "movielens": self.validate_movielens(),
            "tmdb": self.validate_tmdb(),
            "wikipedia": self.validate_wikipedia(),
            "cross_dataset": self.validate_cross_dataset()
        }
        
        logger.info("Dataset validation suite finished.")
        return self.reports

    def validate_imdb(self) -> Dict[str, Any]:
        """Runs data quality checks on the IMDb dataset."""
        logger.info("[Validation] Running IMDb checks...")
        issues = []
        stats = {}
        
        basics = self.datasets.get("imdb", {}).get("title.basics.tsv.gz")
        ratings = self.datasets.get("imdb", {}).get("title.ratings.tsv.gz")
        principals = self.datasets.get("imdb", {}).get("title.principals.tsv.gz")
        crew = self.datasets.get("imdb", {}).get("title.crew.tsv.gz")
        names = self.datasets.get("imdb", {}).get("name.basics.tsv.gz")

        if basics is not None:
            # Row count
            row_cnt = basics.select(pl.len()).collect().item()
            stats["title_basics_rows"] = row_cnt
            
            # 1. Null primary keys
            null_tconst = basics.filter(pl.col("tconst").is_null()).select(pl.len()).collect().item()
            if null_tconst > 0:
                issues.append({
                    "rule": "IMDb primary key cannot be null",
                    "severity": "CRITICAL",
                    "file": "title.basics.tsv.gz",
                    "affected_rows": null_tconst
                })

            # 2. Duplicate primary keys
            dup_tconst = row_cnt - basics.unique(subset=["tconst"]).select(pl.len()).collect().item()
            if dup_tconst > 0:
                issues.append({
                    "rule": "IMDb primary key must be unique",
                    "severity": "CRITICAL",
                    "file": "title.basics.tsv.gz",
                    "affected_rows": dup_tconst
                })

            # 3. Invalid identifier formats (tconst must match ^tt\d+$)
            invalid_ids = basics.filter(~pl.col("tconst").str.contains(r"^tt\d+$")).select(pl.len()).collect().item()
            if invalid_ids > 0:
                issues.append({
                    "rule": r"IMDb identifier format (tconst must match ^tt\d+$)",
                    "severity": "HIGH",
                    "file": "title.basics.tsv.gz",
                    "affected_rows": invalid_ids
                })


            # 4. Malformed dates/years (startYear should be a 4-digit number or null)
            # Polars contains check using regex
            malformed_start_years = basics.filter(
                pl.col("startYear").is_not_null() & ~pl.col("startYear").cast(pl.String).str.contains(r"^\d{4}$")
            ).select(pl.len()).collect().item()
            if malformed_start_years > 0:
                issues.append({
                    "rule": "startYear must be a valid 4-digit integer",
                    "severity": "MEDIUM",
                    "file": "title.basics.tsv.gz",
                    "affected_rows": malformed_start_years
                })

        if names is not None:
            name_row_cnt = names.select(pl.len()).collect().item()
            stats["name_basics_rows"] = name_row_cnt
            
            # Null/Duplicate check
            null_nconst = names.filter(pl.col("nconst").is_null()).select(pl.len()).collect().item()
            dup_nconst = name_row_cnt - names.unique(subset=["nconst"]).select(pl.len()).collect().item()
            if null_nconst > 0 or dup_nconst > 0:
                issues.append({
                    "rule": "IMDb name basics primary key check (null or duplicates)",
                    "severity": "CRITICAL",
                    "file": "name.basics.tsv.gz",
                    "details": f"Null: {null_nconst}, Duplicates: {dup_nconst}"
                })

        # 5. Out-of-bounds ratings
        if ratings is not None:
            invalid_ratings = ratings.filter(
                (pl.col("averageRating") < 0.0) | (pl.col("averageRating") > 10.0)
            ).select(pl.len()).collect().item()
            if invalid_ratings > 0:
                issues.append({
                    "rule": "Ratings must be between 0.0 and 10.0",
                    "severity": "HIGH",
                    "file": "title.ratings.tsv.gz",
                    "affected_rows": invalid_ratings
                })

        # 6. Broken Foreign Keys (Internal)
        if basics is not None:
            # ratings -> basics
            if ratings is not None:
                broken_ratings = ratings.join(basics, on="tconst", how="anti").select(pl.len()).collect().item()
                stats["broken_ratings_fks"] = broken_ratings
                if broken_ratings > 0:
                    issues.append({
                        "rule": "Ratings key tconst must exist in basics",
                        "severity": "HIGH",
                        "file": "title.ratings.tsv.gz",
                        "affected_rows": broken_ratings
                    })
            # principals -> basics
            if principals is not None:
                broken_principals_title = principals.join(basics, on="tconst", how="anti").select(pl.len()).collect().item()
                stats["broken_principals_title_fks"] = broken_principals_title
                if broken_principals_title > 0:
                    issues.append({
                        "rule": "Principals key tconst must exist in basics",
                        "severity": "HIGH",
                        "file": "title.principals.tsv.gz",
                        "affected_rows": broken_principals_title
                    })
                # principals -> names
                if names is not None:
                    broken_principals_name = principals.join(names, on="nconst", how="anti").select(pl.len()).collect().item()
                    stats["broken_principals_name_fks"] = broken_principals_name
                    if broken_principals_name > 0:
                        issues.append({
                            "rule": "Principals key nconst must exist in names",
                            "severity": "HIGH",
                            "file": "title.principals.tsv.gz",
                            "affected_rows": broken_principals_name
                        })
            # crew -> basics
            if crew is not None:
                broken_crew = crew.join(basics, on="tconst", how="anti").select(pl.len()).collect().item()
                stats["broken_crew_fks"] = broken_crew
                if broken_crew > 0:
                    issues.append({
                        "rule": "Crew key tconst must exist in basics",
                        "severity": "HIGH",
                        "file": "title.crew.tsv.gz",
                        "affected_rows": broken_crew
                    })

        return {
            "dataset": "IMDb",
            "validation_status": "FAILED" if any(i["severity"] == "CRITICAL" for i in issues) else "PASSED",
            "issues": issues,
            "statistics": stats
        }

    def validate_movielens(self) -> Dict[str, Any]:
        """Runs data quality checks on the MovieLens dataset."""
        logger.info("[Validation] Running MovieLens checks...")
        issues = []
        stats = {}
        
        movies = self.datasets.get("movielens", {}).get("movies.csv")
        ratings = self.datasets.get("movielens", {}).get("ratings.csv")
        tags = self.datasets.get("movielens", {}).get("tags.csv")
        links = self.datasets.get("movielens", {}).get("links.csv")

        if movies is not None:
            movie_cnt = movies.select(pl.len()).collect().item()
            stats["movies_rows"] = movie_cnt
            # Null or duplicates
            null_id = movies.filter(pl.col("movieId").is_null()).select(pl.len()).collect().item()
            dup_id = movie_cnt - movies.unique(subset=["movieId"]).select(pl.len()).collect().item()
            if null_id > 0 or dup_id > 0:
                issues.append({
                    "rule": "MovieLens movieId validation (null or duplicates)",
                    "severity": "CRITICAL",
                    "file": "movies.csv",
                    "details": f"Null: {null_id}, Duplicates: {dup_id}"
                })

        # Ratings out of bounds (MovieLens is typically 0.5 to 5.0)
        if ratings is not None:
            invalid_ratings = ratings.filter(
                (pl.col("rating") < 0.5) | (pl.col("rating") > 5.0)
            ).select(pl.len()).collect().item()
            stats["ratings_rows"] = ratings.select(pl.len()).collect().item()
            if invalid_ratings > 0:
                issues.append({
                    "rule": "MovieLens ratings must be between 0.5 and 5.0",
                    "severity": "HIGH",
                    "file": "ratings.csv",
                    "affected_rows": invalid_ratings
                })

        # Internal broken foreign keys
        if movies is not None:
            if ratings is not None:
                broken_ratings = ratings.join(movies, on="movieId", how="anti").select(pl.len()).collect().item()
                if broken_ratings > 0:
                    issues.append({
                        "rule": "Ratings movieId must exist in movies.csv",
                        "severity": "HIGH",
                        "file": "ratings.csv",
                        "affected_rows": broken_ratings
                    })
            if tags is not None:
                broken_tags = tags.join(movies, on="movieId", how="anti").select(pl.len()).collect().item()
                if broken_tags > 0:
                    issues.append({
                        "rule": "Tags movieId must exist in movies.csv",
                        "severity": "HIGH",
                        "file": "tags.csv",
                        "affected_rows": broken_tags
                    })
            if links is not None:
                broken_links = links.join(movies, on="movieId", how="anti").select(pl.len()).collect().item()
                if broken_links > 0:
                    issues.append({
                        "rule": "Links movieId must exist in movies.csv",
                        "severity": "HIGH",
                        "file": "links.csv",
                        "affected_rows": broken_links
                    })

        return {
            "dataset": "MovieLens",
            "validation_status": "FAILED" if any(i["severity"] == "CRITICAL" for i in issues) else "PASSED",
            "issues": issues,
            "statistics": stats
        }

    def validate_tmdb(self) -> Dict[str, Any]:
        """Runs data quality checks on the TMDb dataset."""
        logger.info("[Validation] Running TMDb checks...")
        issues = []
        stats = {}
        
        meta = self.datasets.get("tmdb", {}).get("movies_metadata.csv")
        credits = self.datasets.get("tmdb", {}).get("credits.csv")
        keywords = self.datasets.get("tmdb", {}).get("keywords.csv")
        links = self.datasets.get("tmdb", {}).get("links.csv")
        numeric_meta = None

        if meta is not None:
            row_cnt = meta.select(pl.len()).collect().item()
            stats["metadata_rows"] = row_cnt
            
            # Check for non-numeric IDs in movies_metadata
            non_numeric_ids = meta.filter(
                ~pl.col("id").cast(pl.String).str.contains(r"^\d+$")
            ).select(pl.len()).collect().item()
            if non_numeric_ids > 0:
                issues.append({
                    "rule": "TMDb metadata id must be a numeric integer",
                    "severity": "HIGH",
                    "file": "movies_metadata.csv",
                    "affected_rows": non_numeric_ids
                })

            # Check duplicate primary keys (using only numeric rows to be accurate)
            numeric_meta = meta.filter(pl.col("id").cast(pl.String).str.contains(r"^\d+$")).with_columns(
                pl.col("id").cast(pl.Int64)
            )
            num_cnt = numeric_meta.select(pl.len()).collect().item()
            dup_id = num_cnt - numeric_meta.unique(subset=["id"]).select(pl.len()).collect().item()
            if dup_id > 0:
                issues.append({
                    "rule": "TMDb metadata primary key must be unique",
                    "severity": "HIGH",
                    "file": "movies_metadata.csv",
                    "affected_rows": dup_id
                })

            # Malformed dates (should match YYYY-MM-DD or be null)
            malformed_dates = meta.filter(
                pl.col("release_date").is_not_null() & 
                (pl.col("release_date") != "") &
                ~pl.col("release_date").cast(pl.String).str.contains(r"^\d{4}-\d{2}-\d{2}$")
            ).select(pl.len()).collect().item()
            if malformed_dates > 0:
                issues.append({
                    "rule": "TMDb release_date format must be YYYY-MM-DD",
                    "severity": "MEDIUM",
                    "file": "movies_metadata.csv",
                    "affected_rows": malformed_dates
                })

        # Internal FKey checks (Credits/Keywords link to Metadata)
        if meta is not None and numeric_meta is not None:
            if credits is not None:
                # In credits.csv, 'id' corresponds to TMDb movie id
                broken_credits = credits.filter(pl.col("id").cast(pl.String).str.contains(r"^\d+$")).with_columns(
                    pl.col("id").cast(pl.Int64)
                ).join(numeric_meta, on="id", how="anti").select(pl.len()).collect().item()
                stats["broken_credits_fks"] = broken_credits
                if broken_credits > 0:
                    issues.append({
                        "rule": "Credits id must exist in movies_metadata",
                        "severity": "MEDIUM",
                        "file": "credits.csv",
                        "affected_rows": broken_credits
                    })

            if keywords is not None:
                # In keywords.csv, 'id' corresponds to TMDb movie id
                broken_keywords = keywords.filter(pl.col("id").cast(pl.String).str.contains(r"^\d+$")).with_columns(
                    pl.col("id").cast(pl.Int64)
                ).join(numeric_meta, on="id", how="anti").select(pl.len()).collect().item()
                stats["broken_keywords_fks"] = broken_keywords
                if broken_keywords > 0:
                    issues.append({
                        "rule": "Keywords id must exist in movies_metadata",
                        "severity": "MEDIUM",
                        "file": "keywords.csv",
                        "affected_rows": broken_keywords
                    })

        return {
            "dataset": "TMDb",
            "validation_status": "FAILED" if any(i["severity"] == "CRITICAL" for i in issues) else "PASSED",
            "issues": issues,
            "statistics": stats
        }

    def validate_wikipedia(self) -> Dict[str, Any]:
        """Runs data quality checks on the Wikipedia movie plots dataset."""
        logger.info("[Validation] Running Wikipedia checks...")
        issues = []
        stats = {}
        
        wiki = self.datasets.get("wikipedia", {}).get("wiki_movie_plots_deduped.csv")
        if wiki is not None:
            row_cnt = wiki.select(pl.len()).collect().item()
            stats["wiki_rows"] = row_cnt
            
            # 1. Null/Empty plot fields
            empty_plots = wiki.filter(
                pl.col("Plot").is_null() | (pl.col("Plot") == "")
            ).select(pl.len()).collect().item()
            if empty_plots > 0:
                issues.append({
                    "rule": "Wikipedia Plot field cannot be empty",
                    "severity": "HIGH",
                    "file": "wiki_movie_plots_deduped.csv",
                    "affected_rows": empty_plots
                })
                
            # 2. Invalid release years
            malformed_years = wiki.filter(
                ~pl.col("Release Year").cast(pl.String).str.contains(r"^\d{4}$")
            ).select(pl.len()).collect().item()
            if malformed_years > 0:
                issues.append({
                    "rule": "Wikipedia Release Year must be 4-digit integer",
                    "severity": "MEDIUM",
                    "file": "wiki_movie_plots_deduped.csv",
                    "affected_rows": malformed_years
                })

        return {
            "dataset": "Wikipedia",
            "validation_status": "FAILED" if any(i["severity"] == "CRITICAL" for i in issues) else "PASSED",
            "issues": issues,
            "statistics": stats
        }

    def validate_cross_dataset(self) -> Dict[str, Any]:
        """Validates reference integrity links across different dataset providers."""
        logger.info("[Validation] Running cross-dataset link checks...")
        issues = []
        stats = {}

        # 1. MovieLens links.csv referencing IMDb and TMDb
        ml_links = self.datasets.get("movielens", {}).get("links.csv")
        imdb_basics = self.datasets.get("imdb", {}).get("title.basics.tsv.gz")
        tmdb_meta = self.datasets.get("tmdb", {}).get("movies_metadata.csv")

        if ml_links is not None:
            total_links = ml_links.select(pl.len()).collect().item()
            stats["total_movielens_links"] = total_links

            # Validate IMDb connections
            if imdb_basics is not None:
                # Format MovieLens imdbId (int) into tt0114709 format
                # MovieLens has imdbId as integers. Pad to 7 digits (pad to 7 characters,pad left with '0')
                ml_formatted = ml_links.filter(pl.col("imdbId").is_not_null()).with_columns(
                    pl.format("tt{}", pl.col("imdbId").cast(pl.String).str.zfill(7)).alias("tconst")
                )

                
                # Check how many are missing in IMDb title basics
                broken_imdb_links = ml_formatted.join(imdb_basics, on="tconst", how="anti").select(pl.len()).collect().item()
                stats["broken_imdb_links"] = broken_imdb_links
                match_pct = ((total_links - broken_imdb_links) / total_links) * 100 if total_links > 0 else 0
                stats["imdb_link_match_percentage"] = round(match_pct, 2)
                
                if broken_imdb_links > 0:
                    issues.append({
                        "rule": "MovieLens links.csv imdbId should reference a valid IMDb tconst",
                        "severity": "MEDIUM",
                        "details": f"Broken links: {broken_imdb_links} / {total_links} ({match_pct:.2f}% match)"
                    })

            # Validate TMDb connections
            if tmdb_meta is not None:
                # Filter out messy IDs in TMDb meta
                clean_tmdb = tmdb_meta.filter(
                    pl.col("id").cast(pl.String).str.contains(r"^\d+$")
                ).with_columns(
                    pl.col("id").cast(pl.Int64).alias("tmdbId")
                )
                
                broken_tmdb_links = ml_links.filter(pl.col("tmdbId").is_not_null()).with_columns(
                    pl.col("tmdbId").cast(pl.Int64)
                ).join(clean_tmdb, on="tmdbId", how="anti").select(pl.len()).collect().item()
                
                stats["broken_tmdb_links"] = broken_tmdb_links
                match_pct = ((total_links - broken_tmdb_links) / total_links) * 100 if total_links > 0 else 0
                stats["tmdb_link_match_percentage"] = round(match_pct, 2)
                
                if broken_tmdb_links > 0:
                    issues.append({
                        "rule": "MovieLens links.csv tmdbId should reference a valid TMDb id",
                        "severity": "MEDIUM",
                        "details": f"Broken links: {broken_tmdb_links} / {total_links} ({match_pct:.2f}% match)"
                    })

        return {
            "dataset": "Cross-Dataset Links",
            "validation_status": "PASSED",  # Cross-dataset references are advisory (not critical for dataset boundaries)
            "issues": issues,
            "statistics": stats
        }
