import json
import requests

# Load exactly what the code reads
with open("config/settings.json") as f:
    config = json.load(f)

token   = config["telegram"]["bot_token"]
chat_id = config["telegram"]["chat_id"]

print(f"Token length : {len(token)}")
print(f"Token value  : '{token}'")
print(f"Chat ID      : '{chat_id}'")
print()

# Test the token
url = f"https://api.telegram.org/bot{token}/getMe"
print(f"Calling: {url}")
r = requests.get(url, timeout=10)
print(f"Status: {r.status_code}")
print(f"Response: {r.text}")