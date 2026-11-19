
from flask import Blueprint, request, jsonify, render_template

from src.predict import SentimentPredictor
from database.db import SentimentDatabase


# Create Flask blueprint
main = Blueprint("main", __name__)

# Initialize shared components once
predictor = SentimentPredictor()
database = SentimentDatabase()

VALID_SENTIMENTS = {"Negative", "Neutral", "Positive"}


@main.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@main.route("/predict", methods=["POST"])
def predict():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "JSON request body is required."}), 400

    text = data.get("text")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Please provide valid text."}), 400

    try:
        # Generate sentiment prediction
        result = predictor.predict(text)

        # Persist prediction
        prediction_id = database.save_prediction(text, result)

        return jsonify({
            "prediction_id": prediction_id,
            "text": text,
            **result
        }), 200

    except Exception:
        return jsonify({
            "error": "Unable to process the prediction."
        }), 500


@main.route("/feedback", methods=["POST"])
def feedback():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "JSON request body is required."}), 400

    prediction_id = data.get("prediction_id")
    actual_sentiment = data.get("actual_sentiment")

    if not isinstance(prediction_id, int) or isinstance(prediction_id, bool) or prediction_id <= 0:
        return jsonify({"error": "A valid prediction ID is required."}), 400

    if actual_sentiment not in VALID_SENTIMENTS:
        return jsonify({
            "error": "Sentiment must be Negative, Neutral, or Positive."
        }), 400

    try:
        feedback_result = database.save_feedback(
            prediction_id,
            actual_sentiment
        )

        return jsonify(feedback_result), 200

    except ValueError as error:
        return jsonify({"error": str(error)}), 404

    except Exception:
        return jsonify({
            "error": "Unable to save feedback."
        }), 500
