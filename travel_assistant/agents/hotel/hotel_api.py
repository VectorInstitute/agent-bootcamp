"""Implements event retrieval tool for hotel using Amadeus."""

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
            headers={"Authorization": f"Bearer {self.access_token}",
                        "Content-Type": "application/json"}
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
        Search for hotels using the amadeus API.

        Args:
            cityCode (Optional[str]): city code for city name which hotels are located in.

        Returns:
            list[dict]: A list of hotel dictionaries matching the search criteria.

        Raises:
            httpx.HTTPStatusError: If the API request fails.

        Example:
            events = await client.search_events(cityCode="PAR")
        """
        params = {
            "cityCode": cityCode,
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        resp = await self.client.get(url=f"{self.BASE_URL}{self.ENDPOINT}", params=params)
        print("print: ", resp.url)
        resp.raise_for_status()
        return resp.json()
