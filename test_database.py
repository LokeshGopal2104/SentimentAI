
from database.db import SentimentDatabase


# Initialize database
database = SentimentDatabase()

# Sample prediction
sample_result = {
    "sentiment": "Positive",
    "confidence": 0.98,
    "probabilities": {
        "Negative": 0.01,
        "Neutral": 0.01,
        "Positive": 0.98
    }
}

# Save prediction
prediction_id = database.save_prediction(
    "I absolutely love this product!",
    sample_result
)

print("Prediction saved with ID:", prediction_id)

# Submit corrected feedback
feedback = database.save_feedback(
    prediction_id,
    "Negative"
)

print("\nFeedback result:")
print(feedback)

# Retrieve complete record
record = database.get_prediction(prediction_id)

print("\nDatabase record:")
print(record)
