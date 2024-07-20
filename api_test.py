import requests
import json

# URL of your Flask app
url = 'http://localhost:5000/predict_price'

# Sample data to send
data = {
    'flight_id': 'E1247',
    'seat_type': 'Economy',
    'remaining_seats': 18
}

# Send POST request
response = requests.post(url, json=data)

# Check the response
if response.status_code == 200:
    result = response.json()
    print(f"Predicted price: {result['predicted_price']}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)