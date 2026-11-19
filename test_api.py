
import requests


BASE_URL = "http://127.0.0.1:5000"


# Test sentiment prediction
prediction_response = requests.post(
    f"{BASE_URL}/predict",
    json={
        "text": "I absolutely love this product!"
    }
)

print("PREDICTION RESPONSE")
print(prediction_response.status_code)
print(prediction_response.json())


# Extract prediction ID
prediction_data = prediction_response.json()

prediction_id = prediction_data["prediction_id"]


# Test user feedback
feedback_response = requests.post(
    f"{BASE_URL}/feedback",
    json={
        "prediction_id": prediction_id,
        "actual_sentiment": "Positive"
    }
)

print("\nFEEDBACK RESPONSE")
print(feedback_response.status_code)
print(feedback_response.json())
