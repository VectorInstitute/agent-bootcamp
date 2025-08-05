# Replace this with your actual OAuth token
token="SU7lc8v5aB_YS9DnqT2AW61cZ-OUrZ0rTRZ9_TW8"

import requests
from pprint import pprint

response = requests.get(
    url="https://api.predicthq.com/v1/events/",
    headers={
      "Authorization": f"Bearer {token}",
      "Accept": "application/json"
    },
    params={
        "q": "taylor swift"
    }
)

pprint(response.json())