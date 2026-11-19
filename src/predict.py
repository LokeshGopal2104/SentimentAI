
import json
import pickle
import numpy as np
import tensorflow as tf

from pathlib import Path
from src.preprocessing import preprocess_text


# Resolve project root independently of the current terminal directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_model.keras"
TOKENIZER_PATH = PROJECT_ROOT / "models" / "tokenizer.pkl"
CONFIG_PATH = PROJECT_ROOT / "models" / "config.json"


class SentimentPredictor:

    def __init__(self):

        # Load trained model
        self.model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        # Load trained tokenizer
        with open(TOKENIZER_PATH, "rb") as file:
            self.tokenizer = pickle.load(file)

        # Load model configuration
        with open(CONFIG_PATH, "r", encoding="utf-8") as file:
            self.config = json.load(file)

        self.max_sequence_length = self.config["max_sequence_length"]

        self.class_mapping = {
            int(key): value
            for key, value in self.config["class_mapping"].items()
        }

        self.num_classes = self.config["num_classes"]

        print("Sentiment prediction engine initialized successfully!")

    def predict(self, text):

        # Step 1: Preprocess input text
        processed_text = preprocess_text(
            text,
            self.tokenizer,
            self.max_sequence_length
        )

        # Step 2: Generate probabilities
        probabilities = self.model.predict(
            processed_text,
            verbose=0
        )[0]

        # Step 3: Identify predicted class
        predicted_class = int(np.argmax(probabilities))

        # Step 4: Get predicted sentiment
        predicted_sentiment = self.class_mapping[predicted_class]

        # Step 5: Get confidence score
        confidence = float(probabilities[predicted_class])

        # Step 6: Prepare probability distribution
        probability_distribution = {
            self.class_mapping[index]: float(probabilities[index])
            for index in range(self.num_classes)
        }

        # Step 7: Return structured prediction
        return {
            "sentiment": predicted_sentiment,
            "confidence": confidence,
            "probabilities": probability_distribution
        }
