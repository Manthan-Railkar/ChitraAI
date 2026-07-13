import httpx
import asyncio

async def test_find_model():
    gemini_key = "AQ.Ab8RN6K_oDgvMKAH5IzUJaJft9SqzXEKq9uZOyXyykggeC00fg"
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params={"key": gemini_key})
        if resp.status_code == 200:
            models_data = resp.json().get("models", [])
            print("--- Available Models ---")
            for m in models_data:
                name = m.get("name", "")
                if "gemini" in name:
                    print(name)
        else:
            print(f"Error: {resp.status_code}")
            print(resp.text)

asyncio.run(test_find_model())
