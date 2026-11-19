
# SentimentAI
### Feedback-Driven Sentiment Analysis Using LSTM and Pretrained Word Embeddings

SentimentAI is an end-to-end deep learning application that classifies text into three sentiment categories: **Positive, Negative, and Neutral**.

The project uses a Long Short-Term Memory (LSTM) neural network with pretrained Twitter GloVe word embeddings to understand the contextual relationships between words and predict the sentiment expressed in a sentence.

Unlike a standalone sentiment classification model, SentimentAI also provides a Flask-based web application, persistent prediction storage, and a user feedback mechanism that records corrections for future model improvement.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Project Objectives](#3-project-objectives)
4. [Technology Stack](#4-technology-stack)
5. [Dataset](#5-dataset)
6. [Data Preprocessing](#6-data-preprocessing)
7. [Exploratory Data Analysis](#7-exploratory-data-analysis)
8. [Model Development](#8-model-development)
9. [Embedding Experimentation](#9-embedding-experimentation)
10. [Model Training](#10-model-training)
11. [Model Evaluation](#11-model-evaluation)
12. [Model Comparison](#12-model-comparison)
13. [Application Architecture](#13-application-architecture)
14. [Project Structure](#14-project-structure)
15. [Backend Implementation](#15-backend-implementation)
16. [Frontend Implementation](#16-frontend-implementation)
17. [Database Design](#17-database-design)
18. [Feedback Mechanism](#18-feedback-mechanism)
19. [Installation and Execution](#19-installation-and-execution)
20. [API Documentation](#20-api-documentation)
21. [Results and Conclusion](#21-results-and-conclusion)
22. [Limitations and Future Improvements](#22-limitations-and-future-improvements)

---

## 1. Project Overview

SentimentAI is a three-class text classification system designed to identify the emotional polarity of user-provided text.

The application accepts a sentence, preprocesses it using a trained tokenizer, converts it into a numerical sequence, and passes it through an LSTM-based neural network.

The model produces a probability distribution across three sentiment classes:

- Negative
- Neutral
- Positive

The application displays the predicted sentiment, confidence score, and individual class probabilities.

Users can also submit feedback indicating whether a prediction was correct. The application stores predictions and feedback in an SQLite database.

### Key Features

- Three-class sentiment classification.
- LSTM-based deep learning architecture.
- Pretrained Twitter GloVe 100-dimensional embeddings.
- Experimental comparison between random and pretrained embeddings.
- Probability-based prediction output.
- Flask REST API.
- Interactive web interface.
- SQLite prediction and feedback storage.
- User feedback collection.

---

## 2. Problem Statement

Understanding customer opinions, product reviews, and social media conversations is important for businesses and organizations.

Manually analyzing large volumes of text is time-consuming and difficult to scale.

Traditional approaches based on keyword matching often fail to capture the semantic relationships between words.

For example:

- "I absolutely love this product!" expresses positive sentiment.
- "This is the worst experience ever." expresses negative sentiment.
- "The product was delivered yesterday." expresses neutral sentiment.

The objective of this project is to develop a deep learning-based sentiment classification system capable of automatically identifying these three categories.

---

## 3. Project Objectives

The primary objectives are:

1. Develop a three-class sentiment classification model using LSTM.
2. Understand the impact of pretrained word embeddings on model performance.
3. Experimentally compare randomly initialized embeddings with pretrained GloVe embeddings.
4. Evaluate models using accuracy, precision, recall, F1-score, and confusion matrices.
5. Build a reusable prediction engine.
6. Develop REST APIs using Flask.
7. Store predictions and user feedback using SQLite.
8. Build an interactive frontend for real-time sentiment analysis.

---

## 4. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | TensorFlow / Keras |
| Neural Network | LSTM |
| Word Embeddings | Twitter GloVe 100d |
| Data Processing | NumPy, Pandas |
| Model Evaluation | Scikit-learn |
| Development Environment | Google Colab |
| Application Development | VS Code |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Database | SQLite |
| API Communication | HTTP / JSON |

---

## 5. Dataset

The project uses the Twitter Sentiment Dataset provided by Saurabh Shahane on Kaggle.

**Dataset:** Twitter Sentiment Dataset

**Source:** https://www.kaggle.com/datasets/saurabhshahane/twitter-sentiment-dataset

The dataset contains tweets with their corresponding sentiment labels.

### Original Dataset Labels

| Original Label | Sentiment |
|---|---|
| -1 | Negative |
| 0 | Neutral |
| 1 | Positive |

### Dataset Distribution

After removing missing values, blank text entries, and duplicate text entries having the same sentiment, the cleaned dataset contained 162,968 records.

| Sentiment | Records |
|---|---:|
| Negative | 35,509 |
| Neutral | 55,210 |
| Positive | 72,249 |
| **Total** | **162,968** |

The target labels were remapped into three consecutive integer classes:

| Model Label | Sentiment |
|---|---|
| 0 | Negative |
| 1 | Neutral |
| 2 | Positive |

---

## 6. Data Preprocessing

The raw dataset cannot be directly passed into an LSTM network because neural networks operate on numerical representations.

The following preprocessing pipeline was implemented.

### Step 1: Data Cleaning

- Removed missing text values.
- Removed missing target labels.
- Removed blank text entries.
- Removed duplicate text records with the same sentiment.
- Converted sentiment labels into numerical classes.

### Step 2: Train, Validation, and Test Split

The dataset was divided using stratified sampling to preserve class distributions.

| Dataset | Percentage | Records |
|---|---:|---:|
| Training | 70% | 114,077 |
| Validation | 15% | 24,445 |
| Testing | 15% | 24,446 |

Random state: 42.

### Step 3: Tokenization

A Keras Tokenizer was used to convert words into integer indices.

Configuration:

- Maximum vocabulary size: 30,000
- Out-of-vocabulary token: `<OOV>`
- Padding token: 0
- OOV token index: 1

The tokenizer was fitted exclusively on the training dataset to prevent vocabulary leakage from validation and test data.

### Step 4: Sequence Padding

Tweets contain different numbers of words. Therefore, all sequences were converted into a fixed length.

Based on the dataset's word-count distribution, a maximum sequence length of 45 was selected.

Configuration:

- Maximum sequence length: 45
- Padding: Post-padding
- Truncation: Post-truncation

Final input shapes:

| Dataset | Shape |
|---|---|
| Training | (114077, 45) |
| Validation | (24445, 45) |
| Testing | (24446, 45) |

---

## 7. Exploratory Data Analysis

Word-count analysis was performed to understand the length distribution of tweets.

| Statistic | Word Count |
|---|---:|
| Mean | 20.08 |
| Median | 18 |
| 99th Percentile | 42 |
| Maximum | 52 |

A sequence length of 45 was selected to retain most of the information while keeping the model input dimensions manageable.

---

## 8. Model Development

An LSTM-based neural network was selected for sentiment classification.

LSTM networks are a type of recurrent neural network designed to learn dependencies in sequential data.

Unlike traditional feedforward neural networks, LSTMs maintain an internal memory state that helps them capture relationships between words across a sequence.

### Model Architecture

The architecture consists of the following layers:

| Layer | Configuration | Output |
|---|---|---|
| Input | Sequence length 45 | (None, 45) |
| Embedding | 30,000 vocabulary, 100 dimensions | (None, 45, 100) |
| LSTM | 64 units | (None, 64) |
| Dropout | 0.3 | (None, 64) |
| Dense | 32 units, ReLU | (None, 32) |
| Output | 3 units, Softmax | (None, 3) |

**Total trainable parameters:** 3,044,419

### Architecture Explanation

**1. Embedding Layer**

Converts each integer token into a dense 100-dimensional vector.

The embedding layer is initialized differently in the two experiments described below.

**2. LSTM Layer**

Processes the embedded word sequence and learns sequential relationships between words.

It uses 64 hidden units to generate a representation of the input sentence.

**3. Dropout Layer**

Applies a dropout rate of 0.3 to reduce overfitting.

**4. Dense Layer**

A fully connected layer with 32 neurons and ReLU activation learns higher-level feature combinations.

**5. Output Layer**

Contains three neurons with Softmax activation.

The output represents the probability distribution across Negative, Neutral, and Positive classes.

---

## 9. Embedding Experimentation

One of the primary experiments in this project was understanding how pretrained word embeddings influence sentiment classification performance.

Two models with identical architectures were trained.

The only experimental difference was the initialization of the embedding layer.

### Experiment 1: Randomly Initialized Embeddings

In this experiment, word vectors were initialized randomly.

The embedding weights were trainable, allowing the model to learn word representations directly from the sentiment classification dataset.

Advantages:

- Embeddings are learned specifically for the target dataset.
- No external pretrained embedding file is required.
- The model can adapt representations to the task.

### Experiment 2: Pretrained Twitter GloVe Embeddings

The second model used pretrained Twitter GloVe 100-dimensional word vectors.

GloVe stands for Global Vectors for Word Representation.

It learns word representations using global word co-occurrence statistics.

The pretrained Twitter vectors provide semantic information learned from a large collection of Twitter text.

The embedding matrix was constructed by matching tokenizer vocabulary words with their corresponding GloVe vectors.

Embedding configuration:

| Parameter | Value |
|---|---:|
| Vocabulary Size | 30,000 |
| Embedding Dimension | 100 |
| Words Found | 22,333 |
| Words Required | 29,998 |
| Vocabulary Coverage | 74.45% |

Words without a matching pretrained vector retained their initialized embedding values.

The embedding layer was trainable, allowing the pretrained representations to be fine-tuned during sentiment classification training.

### Experimental Fairness

Both models used:

- The same dataset.
- The same train-validation-test split.
- The same vocabulary.
- The same maximum sequence length.
- The same LSTM architecture.
- The same optimizer.
- The same loss function.
- The same training configuration.

This allowed us to compare the impact of embedding initialization while keeping the remaining model configuration consistent.

---

## 10. Model Training

Both models were trained using TensorFlow/Keras.

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Evaluation Metric | Accuracy |
| Batch Size | 128 |
| Maximum Epochs | 10 |
| LSTM Units | 64 |
| Dropout | 0.3 |

### Callbacks

Two callbacks were used to improve training behavior.

**Early Stopping**

- Monitors validation loss.
- Patience: 2 epochs.
- Restores the best model weights.

This helps prevent unnecessary training after validation performance stops improving.

**ReduceLROnPlateau**

- Monitors validation loss.
- Reduction factor: 0.5.
- Patience: 1 epoch.
- Minimum learning rate: 0.000001.

This reduces the learning rate when validation loss stops improving.

### Best Validation Performance

| Model | Best Validation Accuracy | Approximate Best Validation Loss |
|---|---:|---:|
| Random Embeddings | 96.6% | 0.121 |
| GloVe Embeddings | 97.1% | 0.100 |

---

## 11. Model Evaluation

Both models were evaluated on the held-out test dataset containing 24,446 records.

The following evaluation metrics were used.

### Evaluation Metrics

**Accuracy**

Measures the proportion of correctly classified observations.

**Precision**

Measures how many predictions belonging to a particular class were actually correct.

**Recall**

Measures how many actual observations belonging to a class were correctly identified.

**F1-Score**

The harmonic mean of precision and recall.

**Confusion Matrix**

Provides a detailed view of correct predictions and misclassifications across all sentiment classes.

### Test Results

#### Random Embedding Model

| Metric | Score |
|---|---:|
| Test Loss | 0.132664 |
| Accuracy | 96.27% |
| Macro Precision | 95.77% |
| Macro Recall | 96.02% |
| Macro F1-Score | 95.89% |
| Weighted F1-Score | 96.28% |

#### GloVe Embedding Model

| Metric | Score |
|---|---:|
| Test Loss | 0.104455 |
| Accuracy | 97.16% |
| Macro Precision | 97.03% |
| Macro Recall | 96.80% |
| Macro F1-Score | 96.91% |
| Weighted F1-Score | 97.16% |

### Class-Wise Performance

#### Random Embedding Model

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Negative | 91.95% | 94.59% | 93.25% |
| Neutral | 98.83% | 96.79% | 97.80% |
| Positive | 96.53% | 96.69% | 96.61% |

#### GloVe Embedding Model

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Negative | 95.49% | 94.95% | 95.22% |
| Neutral | 98.96% | 97.36% | 98.15% |
| Positive | 96.64% | 98.10% | 97.36% |

---

## 12. Model Comparison

The pretrained GloVe model outperformed the randomly initialized embedding model across all reported test metrics.

| Metric | Random Embeddings | GloVe Embeddings |
|---|---:|---:|
| Accuracy | 96.27% | **97.16%** |
| Macro Precision | 95.77% | **97.03%** |
| Macro Recall | 96.02% | **96.80%** |
| Macro F1 | 95.89% | **96.91%** |
| Weighted F1 | 96.28% | **97.16%** |
| Test Loss | 0.132664 | **0.104455** |

### Key Observations

1. GloVe improved test accuracy by approximately 0.90 percentage points.
2. Macro F1-score improved from 95.89% to 96.91%.
3. Negative sentiment classification improved notably.
4. Positive sentiment recall increased to 98.10%.
5. The GloVe model achieved a lower test loss.

The accuracy improvement corresponds to approximately 219 additional correct predictions on the same test dataset.

### Confusion Matrix Observations

The normalized confusion matrix of the GloVe model showed:

| Actual Class | Predicted Negative | Predicted Neutral | Predicted Positive |
|---|---:|---:|---:|
| Negative | 94.95% | 0.75% | 4.30% |
| Neutral | 0.94% | 97.36% | 1.70% |
| Positive | 1.49% | 0.42% | 98.10% |

The most noticeable classification error was negative sentiment being predicted as positive.

This indicates that some negative expressions may be difficult to distinguish based on their learned representations.

Potential reasons include ambiguous wording, sarcasm, and contextual complexity.

These are possible explanations rather than conclusions established by the confusion matrix alone.

### Final Model Selection

The Twitter GloVe embedding model was selected as the final deployment model because it achieved better performance across all reported evaluation metrics.

---

## 13. Application Architecture

SentimentAI follows a modular architecture that separates model inference, preprocessing, API handling, data persistence, and frontend responsibilities.

### High-Level Workflow

```text
                  USER
                    |
                    v
          HTML / CSS / JavaScript
                    |
                    v
                FLASK API
                    |
              POST /predict
                    |
                    v
            INPUT VALIDATION
                    |
                    v
           TEXT PREPROCESSING
                    |
                    v
             KERAS TOKENIZER
                    |
                    v
            SEQUENCE PADDING
                    |
                    v
             TRAINED LSTM
                    |
                    v
           SOFTMAX PROBABILITIES
                    |
                    v
           PREDICTED SENTIMENT
                    |
             +------+------+
             |             |
             v             v
       SQLITE DATABASE   FRONTEND
             |
             v
       USER FEEDBACK
             |
             v
        POST /feedback
             |
             v
       UPDATE DATABASE
```

### Application Workflow

1. A user enters text into the frontend.
2. JavaScript sends the text to the Flask prediction endpoint.
3. Flask validates the request.
4. The prediction engine preprocesses the text.
5. The tokenizer converts words into integer sequences.
6. The sequence is padded to the required length.
7. The trained LSTM model predicts class probabilities.
8. The predicted sentiment and confidence are returned.
9. The prediction is stored in SQLite.
10. The frontend displays the prediction.
11. The user can submit feedback.
12. Feedback is stored against the corresponding prediction record.

---

## 14. Project Structure

```text
SentimentAI/
│
├── models/
│   ├── sentiment_model.keras
│   ├── tokenizer.pkl
│   └── config.json
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── predict.py
│
├── app/
│   ├── __init__.py
│   ├── routes.py
│   │
│   ├── templates/
│   │   └── index.html
│   │
│   └── static/
│       ├── css/
│       │   └── style.css
│       │
│       └── js/
│           └── app.js
│
├── database/
│   ├── __init__.py
│   ├── db.py
│   └── sentimentai.db
│
├── tests/
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

### Important Directories

**models/**

Contains the trained Keras model, tokenizer, and model configuration.

**src/**

Contains the preprocessing logic and prediction engine.

**app/**

Contains Flask routes, HTML templates, CSS, and JavaScript.

**database/**

Contains SQLite database initialization and database operations.

**tests/**

Reserved for automated application and model tests.

---

## 15. Backend Implementation

The backend is implemented using Flask.

It exposes HTTP endpoints that allow the frontend to communicate with the trained model and database.

### Prediction Engine

The prediction engine is implemented in:

`src/predict.py`

Its responsibilities include:

- Loading the trained model.
- Loading the fitted tokenizer.
- Loading model configuration.
- Preprocessing incoming text.
- Generating probability distributions.
- Returning the predicted sentiment.

### Text Preprocessing

Implemented in:

`src/preprocessing.py`

The preprocessing function:

1. Validates the input.
2. Converts text into token sequences.
3. Applies post-padding and post-truncation.
4. Returns a numerical array suitable for model inference.

### Flask Routes

Implemented in:

`app/routes.py`

The application provides three primary routes:

| Endpoint | Method | Responsibility |
|---|---|---|
| `/` | GET | Serves the frontend |
| `/predict` | POST | Generates sentiment predictions |
| `/feedback` | POST | Stores user feedback |

---

## 16. Frontend Implementation

The frontend uses HTML, CSS, and JavaScript.

It provides a simple interface for interacting with the sentiment analysis model.

### Frontend Features

**Text Input**

Allows users to enter text for sentiment analysis.

**Prediction Display**

Displays the predicted sentiment category.

**Confidence Score**

Displays the probability associated with the predicted class.

**Probability Visualization**

Displays individual probabilities for Negative, Neutral, and Positive classes using progress bars.

**Feedback Interface**

Allows users to select the actual sentiment of a prediction.

JavaScript communicates with Flask endpoints using the Fetch API and JSON requests.

---

## 17. Database Design

SQLite is used to persist predictions and user feedback.

The database contains a `predictions` table.

### Predictions Table

| Column | Data Type | Description |
|---|---|---|
| id | INTEGER | Primary key |
| text | TEXT | Original input text |
| predicted_sentiment | TEXT | Predicted sentiment |
| confidence | REAL | Predicted class probability |
| probabilities | TEXT | JSON probability distribution |
| actual_sentiment | TEXT | User-provided sentiment |
| is_correct | INTEGER | Prediction correctness |
| created_at | TEXT | Prediction timestamp |

### Database Operations

The database layer is implemented in:

`database/db.py`

It supports:

- Database initialization.
- Saving predictions.
- Retrieving prediction records.
- Saving feedback.
- Comparing predicted and actual sentiments.

SQLite stores probability distributions as JSON text.

The `is_correct` field is calculated by comparing the predicted sentiment with the user-provided actual sentiment.

---

## 18. Feedback Mechanism

One of the additional features of SentimentAI is its feedback collection mechanism.

After receiving a prediction, users can select the actual sentiment.

For example:

```text
Input:
I absolutely love this product!

Predicted Sentiment:
Positive

User Feedback:
Negative
```

The system compares the prediction with the submitted feedback and records whether the prediction was correct.

### Important Design Consideration

The current feedback mechanism stores user corrections and evaluates prediction correctness.

It does not automatically retrain the LSTM model after every feedback submission.

Feedback is collected as additional labeled data that can support a future controlled retraining pipeline.

Before using collected feedback for training, it should be validated because user-provided labels may be noisy or incorrect.

A future retraining pipeline can:

1. Export feedback records.
2. Validate and clean the collected labels.
3. Merge approved records with the training dataset.
4. Retrain a candidate model.
5. Evaluate the candidate model.
6. Compare it against the current production model.
7. Promote the candidate only if it satisfies predefined evaluation criteria.

---

## 19. Installation and Execution

### Prerequisites

- Python 3.12.5
- Git
- pip
- VS Code (recommended)

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate into the project directory:

```bash
cd SentimentAI
```

### Step 2: Create a Virtual Environment

Windows:

```powershell
python -m venv venv
```

Activate the environment:

```powershell
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Model Artifacts

Make sure the following files exist:

```text
models/
├── sentiment_model.keras
├── tokenizer.pkl
└── config.json
```

These artifacts are required for prediction.

### Step 5: Run the Application

```bash
python app.py
```

### Step 6: Open the Application

Navigate to:

```text
http://127.0.0.1:5000/
```

The SentimentAI application should now be accessible locally.

---

## 20. API Documentation

### 1. Home Endpoint

**Endpoint:** `/`

**Method:** GET

Returns the application frontend.

### 2. Prediction Endpoint

**Endpoint:** `/predict`

**Method:** POST

**Request Body:**

```json
{
    "text": "I absolutely love this product!"
}
```

**Successful Response:**

```json
{
    "prediction_id": 2,
    "text": "I absolutely love this product!",
    "sentiment": "Positive",
    "confidence": 0.9999456405639648,
    "probabilities": {
        "Negative": 0.00001581588367116637,
        "Neutral": 0.00003851106157526374,
        "Positive": 0.9999456405639648
    }
}
```

**HTTP Status:** 200

### 3. Feedback Endpoint

**Endpoint:** `/feedback`

**Method:** POST

**Request Body:**

```json
{
    "prediction_id": 2,
    "actual_sentiment": "Positive"
}
```

**Successful Response:**

```json
{
    "prediction_id": 2,
    "predicted_sentiment": "Positive",
    "actual_sentiment": "Positive",
    "is_correct": true
}
```

**HTTP Status:** 200

### Error Handling

The API handles invalid input and missing prediction records using appropriate HTTP status codes.

| Status Code | Meaning |
|---|---|
| 200 | Successful request |
| 400 | Invalid request |
| 404 | Prediction record not found |
| 500 | Internal server error |

---

## 21. Results and Conclusion

SentimentAI demonstrates the development of an end-to-end deep learning application, starting from raw text data and ending with a functional web application.

Two LSTM models were developed using different embedding initialization strategies.

The experimental results showed that the pretrained Twitter GloVe model achieved better performance than the randomly initialized embedding model.

### Final Model Performance

| Metric | Final Score |
|---|---:|
| Test Accuracy | 97.16% |
| Macro Precision | 97.03% |
| Macro Recall | 96.80% |
| Macro F1-Score | 96.91% |
| Weighted F1-Score | 97.16% |

### Key Learnings

Through this project, the following concepts were implemented and explored:

- Text preprocessing and tokenization.
- Vocabulary construction.
- Sequence padding and truncation.
- Word embedding representations.
- Random versus pretrained embeddings.
- LSTM architecture and sequential learning.
- Model training and regularization.
- Evaluation using classification metrics.
- Confusion matrix analysis.
- Model serialization and inference.
- Flask REST API development.
- SQLite database integration.
- Frontend and backend communication.
- User feedback collection.

### Conclusion

The project demonstrates that pretrained word embeddings can provide useful semantic representations and improve downstream sentiment classification performance.

The final LSTM model achieved 97.16% test accuracy on the selected dataset.

The application also demonstrates how a trained deep learning model can be integrated into a usable software system through REST APIs, a database, and an interactive frontend.

SentimentAI provides a foundation for future improvements such as controlled feedback-driven retraining, model monitoring, and deployment as a publicly accessible application.

---

## 22. Limitations and Future Improvements

Although SentimentAI achieves strong classification performance, several improvements are possible.

### Current Limitations

1. The model is trained on a specific Twitter sentiment dataset, so its performance may vary on other text domains.
2. Sarcasm and complex contextual expressions can be difficult to classify.
3. GloVe vocabulary coverage is 74.45%, so some dataset words do not have pretrained vectors.
4. Softmax confidence values should not be interpreted as guaranteed prediction correctness.
5. User feedback is collected but is not currently used for automatic model retraining.
6. The current SQLite implementation is intended for a relatively small application workload.

### Future Improvements

- Implement controlled feedback-driven model retraining.
- Add an analytics dashboard.
- Introduce model monitoring and performance tracking.
- Evaluate Transformer-based architectures such as DistilBERT.
- Add prediction history and filtering.
- Explore model calibration.
- Deploy the application using a cloud hosting platform.
- Introduce persistent production database storage.

---

## Author

**SentimentAI — End-to-End Deep Learning Project**

Built using Python, TensorFlow, LSTM, Twitter GloVe, Flask, SQLite, HTML, CSS, and JavaScript.
