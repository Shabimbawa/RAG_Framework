import requests

api_key = "YOUR_API_KEY"
symbol = "AAPL"

url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={api_key}"
response = requests.get(url)

print(response.json())