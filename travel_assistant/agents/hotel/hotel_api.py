"""Implements event retrieval tool for PredictHQ."""

import httpx
import os
from typing import Optional
import asyncio
import requests
from dotenv import load_dotenv

class AsyncAmadeusClient:
    BASE_URL = "https://test.api.amadeus.com/v1"
    ENDPOINT = "/reference-data/locations/hotels/by-city"

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = None

        # Endpoint
        url = "https://test.api.amadeus.com/v1/security/oauth2/token"

        # Headers
        headers = {
            'Content_Type': 'application/x-www-form-urlencoded'
        }

        # Body
        data = {
            'grant_type': 'client_credentials',
            'client_id': self.api_key,
            'client_secret': self.api_secret
        }

        # Send POST request 
        response = requests.post(url, headers=headers, data=data)

        # Parse response
        if response.status_code == 200: 
            token_info = response.json()
            self.access_token = token_info['access_token']
        else: 
            print("Failed to retrieve token:", response.status_code)
            print(response.text)

        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self.access_token}"}
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
            "Authorization": f"Bearer {self.access_token}",
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
load_dotenv()

async def main():
    client = AsyncAmadeusClient(api_key=os.environ.get("AMADEUS_API_KEY"),
                                api_secret=os.environ.get("AMADEUS_API_SECRET"))
    result = await client.search_hotel("PAR")
    print(result)

# # Run the async main function
asyncio.run(main())

# curl 'https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city?cityCode=PAR' -H 'Authorization: Bearer h4y62q8Zxo3scvggU1PC8wGgUVCY'

#  url 'https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city?cityCode=PAR'

# https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city?cityCode=PAR
# https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city?cityCode=PAR
# https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city?cityCode=PAR