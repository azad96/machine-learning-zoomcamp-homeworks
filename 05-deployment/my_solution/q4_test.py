import requests

url = "http://localhost:8000/predict"
client = {"job": "student", "duration": 280, "poutcome": "failure"}
result = requests.post(url, json=client).json()

print(result)