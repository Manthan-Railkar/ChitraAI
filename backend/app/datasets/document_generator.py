import polars as pl


def build_document_expr() -> pl.Expr:
    """
    Constructs a native Polars expression to build movie documents at C++ speeds.
    Structures all semantic movie metadata (genres, cast, directors, writers, 
    ratings, franchise, keywords, overview/plot) into a single natural-language document.
    """
    # Choose between plot_summary and overview, prioritizing the longer description
    desc_expr = (
        pl.when(pl.col("plot_summary").fill_null("").str.len_chars() > pl.col("overview").fill_null("").str.len_chars())
        .then(pl.col("plot_summary"))
        .otherwise(pl.col("overview"))
    )

    expr = (
        # Title & Release Year
        pl.lit("Title: ") + pl.col("title") +
        pl.when(pl.col("release_year").is_not_null())
        .then(pl.lit(" (") + pl.col("release_year").cast(pl.String) + pl.lit(")"))
        .otherwise(pl.lit("")) +
        
        # Genres
        pl.when(pl.col("genres").is_not_null() & (pl.col("genres").list.len() > 0))
        .then(pl.lit("\nGenres: ") + pl.col("genres").list.join(", "))
        .otherwise(pl.lit("")) +
        
        # Directors
        pl.when(pl.col("directors").is_not_null() & (pl.col("directors").list.len() > 0))
        .then(pl.lit("\nDirected by: ") + pl.col("directors").list.join(", "))
        .otherwise(pl.lit("")) +
        
        # Writers
        pl.when(pl.col("writers").is_not_null() & (pl.col("writers").list.len() > 0))
        .then(pl.lit("\nWritten by: ") + pl.col("writers").list.join(", "))
        .otherwise(pl.lit("")) +
        
        # Cast (slice to first 10 members)
        pl.when(pl.col("cast").is_not_null() & (pl.col("cast").list.len() > 0))
        .then(pl.lit("\nStarring: ") + pl.col("cast").list.head(10).list.join(", "))
        .otherwise(pl.lit("")) +
        
        # Franchise / Collection
        pl.when(pl.col("collection_name").is_not_null())
        .then(pl.lit("\nFranchise / Collection: ") + pl.col("collection_name"))
        .otherwise(pl.lit("")) +
        
        # Production Companies
        pl.when(pl.col("production_companies").is_not_null() & (pl.col("production_companies").list.len() > 0))
        .then(pl.lit("\nProduction Companies: ") + pl.col("production_companies").list.join(", "))
        .otherwise(pl.lit("")) +
        
        # Languages
        pl.when(pl.col("languages").is_not_null() & (pl.col("languages").list.len() > 0))
        .then(pl.lit("\nLanguages: ") + pl.col("languages").list.join(", "))
        .otherwise(pl.lit("")) +
        
        # Rating
        pl.when(pl.col("rating_value").is_not_null())
        .then(pl.lit("\nRating: ") + pl.col("rating_value").round(1).cast(pl.String) + pl.lit("/10"))
        .otherwise(pl.lit("")) +
        
        # Certification
        pl.when(pl.col("certification").is_not_null())
        .then(pl.lit("\nCertification: ") + pl.col("certification"))
        .otherwise(pl.lit("")) +

        # Streaming Providers
        pl.when(pl.col("streaming_providers").is_not_null() & (pl.col("streaming_providers").list.len() > 0))
        .then(pl.lit("\nStreaming Providers: ") + pl.col("streaming_providers").list.join(", "))
        .otherwise(pl.lit("")) +

        # Keywords / Themes
        pl.when(pl.col("keywords").is_not_null() & (pl.col("keywords").list.len() > 0))
        .then(pl.lit("\nKeywords / Themes: ") + pl.col("keywords").list.join(", "))
        .otherwise(pl.lit("")) +
        
        # Description
        pl.when(desc_expr.is_not_null() & (desc_expr.str.strip_chars().str.len_chars() > 0))
        .then(pl.lit("\nDescription: ") + desc_expr.str.strip_chars())
        .otherwise(pl.lit(""))
    )
    return expr


def generate_knowledge_base_documents(df: pl.DataFrame) -> pl.DataFrame:
    """
    Appends a structured 'document' column to the movie DataFrame
    for embedding pipeline inputs.
    """
    doc_expr = build_document_expr().alias("document")
    return df.with_columns(doc_expr)
