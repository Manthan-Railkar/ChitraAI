import sys
from pathlib import Path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import asyncio
import json
from app.services.tmdb_service import TMDbService

async def main():
    service = TMDbService()
    print("Fetching raw TMDb details for Avengers: Infinity War (299536)...")
    res = await service.fetch_movie_details(299536)
    if res:
        print("Keys returned in response:")
        print(list(res.keys()))
        print("\nimdb_id:", res.get("imdb_id"))
        print("original_language:", res.get("original_language"))
        print("budget:", res.get("budget"))
        print("revenue:", res.get("revenue"))
        print("status:", res.get("status"))
        
        # Check credits
        credits = res.get("credits", {})
        print("credits keys:", list(credits.keys()) if isinstance(credits, dict) else "Not a dict")
        
        # Check videos
        videos = res.get("videos", {})
        print("videos keys:", list(videos.keys()) if isinstance(videos, dict) else "Not a dict")
        
        # Check watch/providers
        providers = res.get("watch/providers", {})
        print("providers keys:", list(providers.keys()) if isinstance(providers, dict) else "Not a dict")
        
        # Check images
        images = res.get("images", {})
        print("images keys:", list(images.keys()) if isinstance(images, dict) else "Not a dict")
    else:
        print("No response from TMDb.")

if __name__ == "__main__":
    asyncio.run(main())
