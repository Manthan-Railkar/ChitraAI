import asyncio
import httpx
import json
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_URL = "http://127.0.0.1:8002"

QUERIES = [
    "Palme d'Or winner slow burn mystery",
    "Studio Ghibli style family safe fantasy",
    "Underrated A24 thriller",
    "Fast paced action movies starring Tom Cruise",
    "Niche mind-bending sci-fi"
]

async def check_query(client, query):
    url = f"{BASE_URL}/api/v1/recommendations/semantic"
    params = {"q": query, "limit": 3}
    try:
        resp = await client.get(url, params=params, timeout=60.0)
        if resp.status_code == 200:
            data = resp.json()
            recs = data.get("recommendations", [])
            print(f"\nQUERY: '{query}'")
            for idx, r in enumerate(recs):
                match_pct = round(r.get("confidence_score", 0.0) * 100, 2)
                print(f"    {idx+1}. {r.get('title')} ({r.get('release_year')}) -> Match: {match_pct}%")
                print(f"       Rating: {r.get('rating')} | Vote Count: {r.get('vote_count')} | Popularity: {r.get('popularity')}")
                print(f"       Reason: {r.get('recommendation_reason')}")
        else:
            print(f"\nQUERY: '{query}' failed with status {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"\nQUERY: '{query}' failed with error: {e}")

async def main():
    async with httpx.AsyncClient() as client:
        for q in QUERIES:
            await check_query(client, q)

if __name__ == "__main__":
    asyncio.run(main())
