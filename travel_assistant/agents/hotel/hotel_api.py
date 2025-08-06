"""Implements event retrieval tool for PredictHQ."""

import httpx
import os
from typing import Optional
import asyncio

class AsyncAmadeusClient:
    BASE_URL = "https://test.api.amadeus.com/v1"
    ENDPOINT = "/reference-data/locations/hotels/by-city"

    # ENDPOINT = "/shopping/hotel_offers_serach"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer Agd2ZKFJG3JvxOM43yzQsBXXoSny"
            }#, "Content-Type": "application/json"}
        )
    async def search_hotel(
        self,
        cityCode, 
        # chainCodes,     
        # checkInDate,
        # checkOutDate,
        # currency,
        # priceRange
    ) -> list[dict]:
        """
        Search for events using the PredictHQ API.

        Args:
            q (Optional[str]): Free text search query for event titles or descriptions.
            category (Optional[str]): Event category (e.g., 'sports', 'concerts').
            country (Optional[str]): 2-letter country code (e.g., 'US', 'GB').
            start (Optional[str]): Start date (inclusive) in 'YYYY-MM-DD' format.
            end (Optional[str]): End date (inclusive) in 'YYYY-MM-DD' format.
            limit (int): Maximum number of results to return (default: 5).

        Returns:
            list[dict]: A list of event dictionaries matching the search criteria.

        Raises:
            httpx.HTTPStatusError: If the API request fails.

        Example:
            events = await client.search_events(q="music", country="US", start="2024-08-01", end="2024-08-31")
        """
        
        # params = {
        #     "cityCode": cityCode,
        # }
        # # Remove None values
        # params = {k: v for k, v in params.items() if v is not None}
        # self.client(veri)
        # resp = await self.client.get(url=f"{self.BASE_URL}{self.ENDPOINT}", params=params)
        # print("print: ", resp.url)
        # resp.raise_for_status()
        # return resp.json().get("results") #resp.json().get("results", [])
    
        endpoint = "/reference-data/locations/hotels/by-city"
        
        params = {
            "cityCode": cityCode
        }

        
        headers = {
            "Authorization": f"Bearer Agd2ZKFJG3JvxOM43yzQsBXXoSny",
            # "x-rapidapi-host": self.host,
            "Content-Type": "application/json"
        }

        # resp = await self.client.get(url=f"{self.BASE_URL}{self.ENDPOINT}", params=params)
        # print("print: ", resp.url)
        # resp.raise_for_status()
        # return resp.json()#.get("results") #resp.json().get("results", [])
    
        
        async with httpx.AsyncClient(verify=False) as client:
            try:
                response = await client.get(f"{self.BASE_URL}{endpoint}", headers=headers, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                print(f"HTTP error occurred: {exc.response.status_code} - {exc.response.text}")
            except Exception as e:
                print(f"An error occurred: {e}")

# Example usage
async def main():
    client = AsyncAmadeusClient(
        api_key="Agd2ZKFJG3JvxOM43yzQsBXXoSny",
        # host="https://test.api.amadeus.com/v1"
    )
    result = await client.search_hotel("PAR")
    print(result)

# # Run the async main function
asyncio.run(main())

# curl 'https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city?cityCode=PAR' -H 'Authorization: Bearer h4y62q8Zxo3scvggU1PC8wGgUVCY'

#  url 'https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city?cityCode=PAR'

# https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city?cityCode=PAR
# https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city?cityCode=PAR
# https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city?cityCode=PAR