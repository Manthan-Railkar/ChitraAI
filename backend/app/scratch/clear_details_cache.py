import sqlite3
import os

db_path = "app/datasets/cache/tmdb_cache.db"
if not os.path.exists(db_path):
    db_path = "backend/app/datasets/cache/tmdb_cache.db"
if os.path.exists(db_path):
    print("Clearing details and discover tables in TMDb Cache Database...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM movie_details_cache;")
    cursor.execute("DELETE FROM discover_cache;")
    conn.commit()
    conn.close()
    print("TMDb details and discover cache cleared successfully.")
else:
    print("Database file not found.")
