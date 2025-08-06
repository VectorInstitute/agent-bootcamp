import httpx
import asyncio
from datetime import date

class RapidApiFlightSearchClient:

    def __init__(self, api_key: str, host: str):
        self.api_key = api_key
        self.host = host
        self.base_url = f"https://{host}"

    async def search_flight(self, departure_id: str, arrival_id: str, outbound_date: str = None, travel_class: str = "ECONOMY",
                            adults: int = 1, children: int = 0, currency_code: str = "CAD", language_code: str = "en-US"):
        
       
        """
        Searches for available flights based on the provided travel parameters.

        Args:
            departure_id (str): IATA code or identifier for the departure location.
            arrival_id (str): IATA code or identifier for the arrival location.
            outbound_date (str): Date of departure in YYYY-MM-DD format.
            travel_class (str, optional): Travel class (e.g., "ECONOMY", "BUSINESS"). Defaults to "ECONOMY".
            adults (int, optional): Number of adult passengers. Defaults to 1.
            children (int, optional): Number of child passengers. Defaults to 0.
            currency_code (str, optional): Currency code for pricing (e.g., "CAD"). Defaults to "CAD".
            language_code (str, optional): Language code for the response (e.g., "en-US"). Defaults to "en-US".

        Returns:
            dict: JSON response containing flight search results if successful.

        Raises:
            httpx.HTTPStatusError: If the HTTP request returns an error status.
            Exception: For any other unexpected errors during the request.
        """

        endpoint = "/api/v1/searchFlights"
        if outbound_date is None:
            outbound_date = date.today() + 7
        params = {
            "departure_id": departure_id,
            "arrival_id": arrival_id,
            "outbound_date": outbound_date,
            "travel_class": travel_class,
            "adults": adults,
            "children": children,
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

class RapidApiAirportSearchClient:
    def __init__(self, api_key: str, host: str):
        self.api_key = api_key
        self.host = host
        self.base_url = f"https://{host}"

    async def search_airport(self, query: str, language: str = "en-US", location: str = "US"):
        """
        Searches for airport suggestions based on a partial query string.

        Args:
            query (str): Partial or full name of the airport, city, or location to search.
            language (str, optional): Language code for the response (e.g., "en-US"). Defaults to "en-US".
            location (str, optional): Country code to narrow down the search (e.g., "US"). Defaults to "US".

        Returns:
            dict: JSON response containing a list of matching airports or locations.

        Raises:
            httpx.HTTPStatusError: If the HTTP request returns an error status.
            Exception: For any other unexpected errors during the request.
        """

        endpoint = "/auto-complete"
        
        params = {
            "query": query,
            "language": language,
            "location": location,
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

# # Example usage of flight search
# async def flight_search_test():
#     client = RapidApiFlightSearchClient(
#         api_key="205ec8b542mshe9c9457439a87bcp15fbb8jsn4dd5e4c7d075",
#         host="google-flights2.p.rapidapi.com"
#     )
#     result = await client.search_flight("DEL", "JFK", "2025-08-10")
#     print(result)
# # Run the async main function
# asyncio.run(flight_search_test())

# Example usage of airport search
# async def airport_search_test():
#     client = RapidApiAirportSearchClient(
#         api_key="205ec8b542mshe9c9457439a87bcp15fbb8jsn4dd5e4c7d075",
#         host="google-flights4.p.rapidapi.com"
#     )
#     result = await client.search_airport("Montreal")
#     print(result)
# # Run the async main function
# asyncio.run(airport_search_test())

