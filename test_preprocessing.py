
import json
import pickle

from src.preprocessing import preprocess_text


# Load the trained tokenizer
with open("models/tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# Load model configuration
with open("models/config.json", "r") as file:
    config = json.load(file)

# Sample input
text = "I absolutely love this product!"

# Preprocess
processed_text = preprocess_text(
    text,
    tokenizer,
    config["max_sequence_length"]
)

print("Original text:", text)
print("Processed sequence:", processed_text)
print("Output shape:", processed_text.shape)
print("Data type:", processed_text.dtype)
