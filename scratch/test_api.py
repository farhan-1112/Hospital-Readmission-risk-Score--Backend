import requests
import json

url = "http://127.0.0.1:8000/analyze"
data = {
    "risk_score": 0.85,
    "patient": {
        "age": "[70-80)",
        "gender": "Female",
        "race": "Caucasian",
        "A1Cresult": ">8",
        "number_inpatient": 2,
        "number_emergency": 1,
        "num_medications": 15,
        "time_in_hospital": 5
    }
}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")
