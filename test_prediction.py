
from src.predict import SentimentPredictor


# Initialize prediction engine
predictor = SentimentPredictor()


# Test examples
test_sentences = [
    "I absolutely love this product!",
    "This is the worst experience ever.",
    "The product was delivered yesterday."
]


# Generate predictions
for text in test_sentences:

    result = predictor.predict(text)

    print("\nText:", text)
    print("Predicted Sentiment:", result["sentiment"])
    print("Confidence:", round(result["confidence"] * 100, 2), "%")

    print("Probabilities:")

    for sentiment, probability in result["probabilities"].items():
        print(f"{sentiment}: {probability * 100:.2f}%")
