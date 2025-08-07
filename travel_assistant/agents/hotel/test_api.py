# Replace this with your actual OAuth token
access_token="ACCESS_TOKEN"
import requests
from pprint import pprint

response = requests.get(
    url="https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city",
    headers={
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    },
    params={
        "cityCode": "TOR"
    }
)

pprint(response.json())