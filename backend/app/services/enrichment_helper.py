import asyncio
from typing import Dict, Any, Optional, List
from loguru import logger


from app.services.tmdb_service import TMDbService


async def enrich_movie_with_tmdb(movie: Dict[str, Any], tmdb_service: TMDbService) -> Dict[str, Any]:
    """
    Dynamically enriches a single movie result dictionary with TMDb metadata
    (poster, backdrop, trailer, streaming providers, certification, runtime, cast, popularity).
    Checks cache first, then calls API asynchronously if key is configured.
    """
    tmdb_id = movie.get("tmdb_id")
    imdb_id = movie.get("imdb_id")
    
    # 1. Resolve TMDb ID via IMDb ID if missing
    if not tmdb_id and imdb_id:
        try:
            tmdb_id = await tmdb_service.fetch_tmdb_id_by_imdb(imdb_id)
        except Exception as e:
            logger.debug(f"Failed to resolve TMDb ID for IMDb ID {imdb_id}: {e}")

    if not tmdb_id:
        # Fallback to local DB metadata only
        return movie

    # 2. Fetch full TMDb details (cached automatically by tmdb_service)
    try:
        # Fetch details with credits if possible (credits are appended to check cast)
        # Note: fetch_movie_details uses default config.
        # To get cast/credits, let's check if the cache/API returns it.
        # We can append credits dynamically by calling _make_request if needed, or by checking
        # if the service already appends it.
        # Wait, the default appends are "videos,watch/providers,release_dates,keywords".
        # Let's write a small patch or query specifically with credits appended.
        # Actually, let's call _make_request inside if api_key is present, or just use details!
        # Let's inspect the details returned by fetch_movie_details:
        details = await tmdb_service.fetch_movie_details(int(tmdb_id))
        
        if details:
            # Merge fields from details
            
            # Overview
            if details.get("overview"):
                movie["overview"] = details.get("overview")

            # Popularity
            if details.get("popularity") is not None:
                movie["popularity"] = details.get("popularity")
                
            # Runtime
            if details.get("runtime") is not None:
                movie["runtime_minutes"] = details.get("runtime")
                
            # Poster and Backdrop Paths
            if details.get("poster_path"):
                poster = details.get("poster_path")
                movie["poster_path"] = poster if poster.startswith("http") else f"https://image.tmdb.org/t/p/w500{poster}"
            if details.get("backdrop_path"):
                backdrop = details.get("backdrop_path")
                movie["backdrop_path"] = backdrop if backdrop.startswith("http") else f"https://image.tmdb.org/t/p/w1280{backdrop}"

            # Genres
            genres = [g.get("name") for g in details.get("genres", []) if g.get("name")]
            if genres:
                movie["genres"] = genres

            # Keywords
            kw_data = details.get("keywords", {})
            if isinstance(kw_data, dict):
                kw_list = kw_data.get("keywords", [])
                if not kw_list:
                    kw_list = kw_data.get("results", [])
            else:
                kw_list = kw_data  # Already a list in some responses
            
            keywords = [kw.get("name") for kw in kw_list if isinstance(kw, dict) and kw.get("name")]
            if keywords:
                movie["keywords"] = keywords



            
            # US Streaming Providers
            providers = details.get("watch/providers", {}).get("results", {}).get("US", {}).get("flatrate", [])
            if providers:
                movie["streaming_providers"] = [p.get("provider_name") for p in providers if p.get("provider_name")]
                
            # US Certification
            certification = None
            releases = details.get("release_dates", {}).get("results", [])
            for r in releases:
                if r.get("iso_3166_1") == "US":
                    dates = r.get("release_dates", [])
                    for d in dates:
                        if d.get("certification"):
                            certification = d.get("certification")
                            break
                    if certification:
                        break
            if certification:
                movie["certification"] = certification

            # Cast Extraction (credits block check)
            credits = details.get("credits", {})
            if credits:
                cast = [member.get("name") for member in credits.get("cast", [])[:10] if member.get("name")]
                if cast:
                    movie["cast"] = cast
                    
    except Exception as e:
        logger.warning(f"Error enriching movie ID {tmdb_id} with TMDb API details: {e}")

    return movie


async def enrich_movie_list(movies: List[Dict[str, Any]], tmdb_service: TMDbService) -> List[Dict[str, Any]]:
    """Enriches a list of movies concurrently using asyncio.gather."""
    if not movies:
        return []
    
    tasks = [enrich_movie_with_tmdb(m, tmdb_service) for m in movies]
    enriched = await asyncio.gather(*tasks)
    return list(enriched)
