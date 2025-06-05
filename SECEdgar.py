import requests

headers = {
    "User-Agent": "rhenzlargo80@gmail.com"
}

url = "https://data.sec.gov/submissions/CIK0000320193.json"
response = requests.get(url, headers=headers)
print(response.json())



