import sys
import unittest
from pathlib import Path
import polars as pl

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.datasets.document_generator import build_document_expr, generate_knowledge_base_documents
from app.datasets.adapters import CANONICAL_SCHEMA, CANONICAL_COLUMNS


class TestDocumentGenerator(unittest.TestCase):
    def test_document_generation_complete(self):
        """Verifies that all fields are correctly formatted in a fully populated record."""
        df = pl.DataFrame({
            "title": ["Toy Story"],
            "release_year": [1995],
            "genres": [["Animation", "Comedy"]],
            "directors": [["John Lasseter"]],
            "writers": [["Joss Whedon", "Andrew Stanton"]],
            "cast": [["Tom Hanks", "Tim Allen", "Don Rickles", "Jim Varney", "Wallace Shawn", "John Ratzenberger", "Annie Potts", "John Morris", "Erik von Detten", "Laurie Metcalf", "R. Lee Ermey"]], # 11 actors
            "collection_name": ["Toy Story Collection"],
            "production_companies": [["Pixar", "Walt Disney Pictures"]],
            "languages": [["English", "Spanish"]],
            "rating_value": [8.3],
            "certification": ["G"],
            "streaming_providers": [["Disney Plus"]],
            "keywords": [["toys", "friendship", "rivalry"]],
            "overview": ["A toy story overview."],
            "plot_summary": ["Toy story detailed plot description containing more words than overview."]
        })

        # Add remaining schema columns
        for col, dtype in CANONICAL_SCHEMA.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
        df = df.cast(CANONICAL_SCHEMA)

        # Generate document
        enriched = generate_knowledge_base_documents(df)
        self.assertIn("document", enriched.columns)

        doc = enriched.select("document").to_series().to_list()[0]
        
        # Verify title & release year
        self.assertIn("Title: Toy Story (1995)", doc)
        
        # Verify lists joined by comma
        self.assertIn("Genres: Animation, Comedy", doc)
        self.assertIn("Directed by: John Lasseter", doc)
        self.assertIn("Written by: Joss Whedon, Andrew Stanton", doc)
        
        # Verify cast sliced to 10
        self.assertIn("Starring: Tom Hanks, Tim Allen, Don Rickles, Jim Varney, Wallace Shawn, John Ratzenberger, Annie Potts, John Morris, Erik von Detten, Laurie Metcalf", doc)
        self.assertNotIn("R. Lee Ermey", doc)  # 11th actor sliced out
        
        # Verify new fields
        self.assertIn("Franchise / Collection: Toy Story Collection", doc)
        self.assertIn("Production Companies: Pixar, Walt Disney Pictures", doc)
        self.assertIn("Languages: English, Spanish", doc)
        self.assertIn("Rating: 8.3/10", doc)
        self.assertIn("Certification: G", doc)
        self.assertIn("Streaming Providers: Disney Plus", doc)
        self.assertIn("Keywords / Themes: toys, friendship, rivalry", doc)
        
        # Verify description prioritization (plot_summary is longer than overview)
        self.assertIn("Description: Toy story detailed plot description containing more words than overview.", doc)
        self.assertNotIn("A toy story overview.", doc)

    def test_document_generation_partial(self):
        """Verifies that missing fields are gracefully omitted from the document without making it null."""
        df = pl.DataFrame({
            "title": ["Inception"],
            "release_year": [2010],
            "genres": [["Action", "Sci-Fi"]],
            "directors": [None],
            "writers": [None],
            "cast": [None],
            "collection_name": [None],
            "production_companies": [None],
            "languages": [None],
            "rating_value": [None],
            "certification": [None],
            "streaming_providers": [None],
            "keywords": [None],
            "overview": ["A thief who steals corporate secrets through the use of dream-sharing technology."],
            "plot_summary": [None]
        })

        # Add remaining schema columns
        for col, dtype in CANONICAL_SCHEMA.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
        df = df.cast(CANONICAL_SCHEMA)

        # Generate document
        enriched = generate_knowledge_base_documents(df)
        doc = enriched.select("document").to_series().to_list()[0]

        self.assertIn("Title: Inception (2010)", doc)
        self.assertIn("Genres: Action, Sci-Fi", doc)
        self.assertNotIn("Directed by:", doc)
        self.assertNotIn("Written by:", doc)
        self.assertNotIn("Starring:", doc)
        self.assertNotIn("Franchise / Collection:", doc)
        self.assertNotIn("Rating:", doc)
        self.assertIn("Description: A thief who steals corporate secrets through the use of dream-sharing technology.", doc)


if __name__ == "__main__":
    unittest.main()
