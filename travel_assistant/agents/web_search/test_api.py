"""Simple script to test DuckDuckGo API."""

from pprint import pprint

import requests


response = requests.get(
    "https://api.duckduckgo.com/",
    params={"q": "openai", "format": "json"},
)

pprint(response.json())
