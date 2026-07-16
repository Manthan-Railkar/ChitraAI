import asyncio
from typing import Dict, Any, Optional, List
from loguru import logger


from app.services.tmdb_service import TMDbService


async def enrich_movie_with_tmdb(movie: Dict[str, Any], tmdb_service: TMDbService) -> Dict[str, Any]:
    """
    Dynamically enriches a single movie result dictionary with TMDb metadata,
    including credits, watch providers, similar movies, recommendations, images, and trailers.
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
        return movie

    # 2. Fetch full TMDb details
    try:
        details = await tmdb_service.fetch_movie_details(int(tmdb_id))
        
        if details:
            # Step 1: Full Movie Enrichment
            movie["tmdb_id"] = details.get("id") or tmdb_id
            movie["imdb_id"] = details.get("imdb_id") or imdb_id
            movie["title"] = details.get("title") or movie.get("title")
            movie["original_title"] = details.get("original_title") or movie.get("original_title")
            movie["overview"] = details.get("overview") or movie.get("overview")
            movie["runtime_minutes"] = details.get("runtime") or movie.get("runtime_minutes")
            movie["release_date"] = details.get("release_date")
            movie["release_year"] = int(details.get("release_date", "0000")[:4]) if details.get("release_date") else movie.get("release_year")
            movie["status"] = details.get("status")
            movie["original_language"] = details.get("original_language")
            movie["production_countries"] = [c.get("name") for c in details.get("production_countries", []) if c.get("name")]
            movie["production_companies"] = [c.get("name") for c in details.get("production_companies", []) if c.get("name")]
            movie["budget"] = details.get("budget")
            movie["revenue"] = details.get("revenue")
            movie["homepage"] = details.get("homepage")
            movie["adult"] = details.get("adult")
            movie["video"] = details.get("video")
            movie["vote_count"] = details.get("vote_count") or movie.get("vote_count")
            movie["rating_value"] = details.get("vote_average") or movie.get("rating_value") or movie.get("rating")
            
            # Collection / Franchise
            col_name = None
            if details.get("belongs_to_collection"):
                col_name = details.get("belongs_to_collection", {}).get("name")
            movie["collection"] = col_name

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
                kw_list = kw_data.get("keywords", []) or kw_data.get("results", [])
            else:
                kw_list = kw_data
            keywords = [kw.get("name") for kw in kw_list if isinstance(kw, dict) and kw.get("name")]
            if keywords:
                movie["keywords"] = keywords

            # Step 2: Credits
            credits = details.get("credits", {})
            if credits:
                movie["cast"] = [member.get("name") for member in credits.get("cast", [])[:10] if member.get("name")]
                movie["directors"] = [member.get("name") for member in credits.get("crew", []) if member.get("job") == "Director" and member.get("name")]
                movie["writers"] = [member.get("name") for member in credits.get("crew", []) if member.get("job") in ["Writer", "Screenplay", "Author"] and member.get("name")]
                
                # Single names for Producer, Composer, Cinematographer
                producer_list = [member.get("name") for member in credits.get("crew", []) if member.get("job") == "Producer" and member.get("name")]
                movie["producer"] = producer_list[0] if producer_list else None
                
                composer_list = [member.get("name") for member in credits.get("crew", []) if member.get("job") in ["Original Music Composer", "Music"] and member.get("name")]
                movie["composer"] = composer_list[0] if composer_list else None
                
                cine_list = [member.get("name") for member in credits.get("crew", []) if member.get("job") in ["Director of Photography", "Cinematography"] and member.get("name")]
                movie["cinematographer"] = cine_list[0] if cine_list else None

            # Step 3: Trailer Support
            videos = details.get("videos", {}).get("results", [])
            trailer = None
            for v in videos:
                if v.get("site") == "YouTube" and v.get("type") == "Trailer" and "official" in v.get("name", "").lower():
                    trailer = v
                    break
            if not trailer:
                for v in videos:
                    if v.get("site") == "YouTube" and v.get("type") == "Trailer":
                        trailer = v
                        break
            if not trailer:
                for v in videos:
                    if v.get("site") == "YouTube" and v.get("type") == "Teaser":
                        trailer = v
                        break
            if trailer:
                movie["trailer_url"] = f"https://www.youtube.com/watch?v={trailer.get('key')}"
                movie["youtube_key"] = trailer.get("key")
                movie["trailer_type"] = trailer.get("type")
                movie["trailer_name"] = trailer.get("name")

            # Step 4: Streaming Providers
            providers_results = details.get("watch/providers", {}).get("results", {})
            flatrate = providers_results.get("US", {}).get("flatrate", []) or providers_results.get("IN", {}).get("flatrate", [])
            if not flatrate:
                for country_code, country_data in providers_results.items():
                    if country_data.get("flatrate"):
                        flatrate = country_data.get("flatrate")
                        break
            movie["streaming_providers"] = [p.get("provider_name") for p in flatrate if p.get("provider_name")]

            # Step 5: Similar Movies
            def map_simplified_movie(m: dict) -> dict:
                poster_path = m.get("poster_path")
                backdrop_path = m.get("backdrop_path")
                return {
                    "tmdb_id": m.get("id"),
                    "title": m.get("title"),
                    "release_year": int(m.get("release_date", "0000")[:4]) if m.get("release_date") else None,
                    "rating": m.get("vote_average"),
                    "poster_path": poster_path if (not poster_path or poster_path.startswith("http")) else f"https://image.tmdb.org/t/p/w500{poster_path}",
                    "backdrop_path": backdrop_path if (not backdrop_path or backdrop_path.startswith("http")) else f"https://image.tmdb.org/t/p/w1280{backdrop_path}"
                }
            similar_movies = details.get("similar", {}).get("results", [])[:5]
            movie["similar_movies"] = [map_simplified_movie(sm) for sm in similar_movies]

            # Step 6: TMDb Recommendations
            recommended_movies = details.get("recommendations", {}).get("results", [])[:5]
            movie["recommended_movies"] = [map_simplified_movie(rm) for rm in recommended_movies]

            # Step 7: Image Assets (Logo URL)
            images = details.get("images", {})
            logos = images.get("logos", [])
            logo_url = None
            if logos:
                logo_path = logos[0].get("file_path")
                logo_url = f"https://image.tmdb.org/t/p/w500{logo_path}"
            movie["logo_url"] = logo_url

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
