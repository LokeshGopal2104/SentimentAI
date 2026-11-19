
import sqlite3
import json

from pathlib import Path
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "database" / "sentimentai.db"


class SentimentDatabase:

    def __init__(self):
        self.database_path = DATABASE_PATH
        self.initialize_database()

    def get_connection(self):
        return sqlite3.connect(self.database_path)

    def initialize_database(self):

        with self.get_connection() as connection:

            cursor = connection.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    text TEXT NOT NULL,

                    predicted_sentiment TEXT NOT NULL,

                    confidence REAL NOT NULL,

                    probabilities TEXT NOT NULL,

                    actual_sentiment TEXT,

                    is_correct INTEGER,

                    created_at TEXT NOT NULL
                )
            """)

            connection.commit()

    def save_prediction(self, text, result):

        with self.get_connection() as connection:

            cursor = connection.cursor()

            cursor.execute("""
                INSERT INTO predictions (
                    text,
                    predicted_sentiment,
                    confidence,
                    probabilities,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
            """, (
                text,
                result["sentiment"],
                result["confidence"],
                json.dumps(result["probabilities"]),
                datetime.now(timezone.utc).isoformat()
            ))

            connection.commit()

            return cursor.lastrowid

    def save_feedback(self, prediction_id, actual_sentiment):

        with self.get_connection() as connection:

            cursor = connection.cursor()

            cursor.execute("""
                SELECT predicted_sentiment
                FROM predictions
                WHERE id = ?
            """, (prediction_id,))

            record = cursor.fetchone()

            if record is None:
                raise ValueError("Prediction record not found.")

            predicted_sentiment = record[0]

            is_correct = int(
                predicted_sentiment == actual_sentiment
            )

            cursor.execute("""
                UPDATE predictions
                SET actual_sentiment = ?,
                    is_correct = ?
                WHERE id = ?
            """, (
                actual_sentiment,
                is_correct,
                prediction_id
            ))

            connection.commit()

            return {
                "prediction_id": prediction_id,
                "predicted_sentiment": predicted_sentiment,
                "actual_sentiment": actual_sentiment,
                "is_correct": bool(is_correct)
            }

    def get_prediction(self, prediction_id):

        with self.get_connection() as connection:

            connection.row_factory = sqlite3.Row

            cursor = connection.cursor()

            cursor.execute("""
                SELECT *
                FROM predictions
                WHERE id = ?
            """, (prediction_id,))

            record = cursor.fetchone()

            if record is None:
                return None

            result = dict(record)

            result["probabilities"] = json.loads(
                result["probabilities"]
            )

            return result
