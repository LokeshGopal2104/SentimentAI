
// ==========================================
// DOM ELEMENTS
// ==========================================

const textInput = document.getElementById("text-input");
const analyzeButton = document.getElementById("analyze-btn");

const resultSection = document.getElementById("result-section");
const sentimentResult = document.getElementById("sentiment-result");
const confidenceResult = document.getElementById("confidence-result");

const errorMessage = document.getElementById("error-message");
const feedbackMessage = document.getElementById("feedback-message");

const feedbackButtons = document.querySelectorAll(".feedback-btn");


// ==========================================
// APPLICATION STATE
// ==========================================

let currentPredictionId = null;


// ==========================================
// PREDICT SENTIMENT
// ==========================================

analyzeButton.addEventListener("click", async () => {

    const text = textInput.value.trim();

    // Validate input
    if (!text) {
        errorMessage.textContent = "Please enter some text.";
        resultSection.classList.add("hidden");
        return;
    }

    errorMessage.textContent = "";
    feedbackMessage.textContent = "";

    analyzeButton.disabled = true;
    analyzeButton.textContent = "Analyzing...";

    try {

        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction failed.");
        }

        // Store prediction ID for feedback
        currentPredictionId = data.prediction_id;

        // Display prediction
        displayPrediction(data);

    } catch (error) {

        errorMessage.textContent = error.message;
        resultSection.classList.add("hidden");

    } finally {

        analyzeButton.disabled = false;
        analyzeButton.textContent = "Analyze Sentiment";

    }

});


// ==========================================
// DISPLAY PREDICTION
// ==========================================

function displayPrediction(data) {

    resultSection.classList.remove("hidden");

    // Display sentiment
    sentimentResult.textContent = data.sentiment;

    sentimentResult.className = data.sentiment.toLowerCase();

    // Display confidence
    confidenceResult.textContent =
        (data.confidence * 100).toFixed(2) + "%";

    // Display class probabilities
    const probabilities = data.probabilities;

    updateProbability(
        "negative",
        probabilities.Negative
    );

    updateProbability(
        "neutral",
        probabilities.Neutral
    );

    updateProbability(
        "positive",
        probabilities.Positive
    );

    // Reset feedback buttons for the new prediction
    feedbackButtons.forEach(button => {
        button.disabled = false;
    });

    feedbackMessage.textContent = "";

    // Scroll to results
    resultSection.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });

}


// ==========================================
// UPDATE PROBABILITY BARS
// ==========================================

function updateProbability(sentiment, probability) {

    const percentage = probability * 100;

    document.getElementById(`${sentiment}-value`)
        .textContent = percentage.toFixed(2) + "%";

    document.getElementById(`${sentiment}-bar`)
        .style.width = percentage + "%";

}


// ==========================================
// SUBMIT USER FEEDBACK
// ==========================================

feedbackButtons.forEach(button => {

    button.addEventListener("click", async () => {

        if (currentPredictionId === null) {
            feedbackMessage.textContent =
                "Please analyze some text first.";
            return;
        }

        const actualSentiment = button.dataset.sentiment;

        // Prevent duplicate submissions while processing
        feedbackButtons.forEach(btn => {
            btn.disabled = true;
        });

        feedbackMessage.textContent = "Submitting feedback...";

        try {

            const response = await fetch("/feedback", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    prediction_id: currentPredictionId,
                    actual_sentiment: actualSentiment
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Feedback submission failed.");
            }

            feedbackMessage.textContent = data.is_correct
                ? "Thank you! Your feedback confirms the prediction."
                : "Thank you! Your correction has been recorded.";

        } catch (error) {

            feedbackMessage.textContent = error.message;

            // Allow retry if submission fails
            feedbackButtons.forEach(btn => {
                btn.disabled = false;
            });

        }

    });

});
