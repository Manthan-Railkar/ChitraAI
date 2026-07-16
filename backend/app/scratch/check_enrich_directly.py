import sys
from pathlib import Path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import asyncio
from app.services.tmdb_service import TMDbService
from app.services.enrichment_helper import enrich_movie_with_tmdb

async def main():
    service = TMDbService()
    # 1. Test Oppenheimer (TMDb ID: 872585)
    movie = {"tmdb_id": 872585}
    res = await enrich_movie_with_tmdb(movie, service)
    print("Enriched Oppenheimer:")
    print("imdb_id:", res.get("imdb_id"))
    print("budget:", res.get("budget"))
    print("revenue:", res.get("revenue"))
    print("logo_url:", res.get("logo_url"))
    print("trailer_url:", res.get("trailer_url"))
    print("similar_movies:", len(res.get("similar_movies", [])))
    print("recommended_movies:", len(res.get("recommended_movies", [])))

if __name__ == "__main__":
    asyncio.run(main())
