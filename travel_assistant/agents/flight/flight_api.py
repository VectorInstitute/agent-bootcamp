import httpx
import asyncio

class RapidApiClient:
    def __init__(self, api_key: str, host: str):
        self.api_key = api_key
        self.host = host
        self.base_url = f"https://{host}"

    async def search_flight(self, departure_id: str, arrival_id: str, outbound_date: str, currency_code: str = "USD", language_code: str = "en-US"):
        endpoint = "/api/v1/getPriceGraph"
        
        params = {
            "departure_id": departure_id,
            "arrival_id": arrival_id,
            "outbound_date": outbound_date,
            "currency_code": currency_code,
            "language_code": language_code
        }

        
        headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": self.host,
            "Content-Type": "application/json"
        }


        
        async with httpx.AsyncClient(verify=False) as client:
            try:
                response = await client.get(f"{self.base_url}{endpoint}", headers=headers, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                print(f"HTTP error occurred: {exc.response.status_code} - {exc.response.text}")
            except Exception as e:
                print(f"An error occurred: {e}")



# Example usage
async def main():
    client = RapidApiClient(
        api_key="205ec8b542mshe9c9457439a87bcp15fbb8jsn4dd5e4c7d075",
        host="google-flights2.p.rapidapi.com"
    )
    result = await client.search_flight("DEL", "JFK", "2025-08-10")
    print(result)

# Run the async main function
asyncio.run(main())
