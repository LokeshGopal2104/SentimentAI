
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


def preprocess_text(text, tokenizer, max_sequence_length):
    """
    Convert raw text into a padded numerical sequence.

    Args:
        text (str): User-provided text.
        tokenizer: Trained Keras tokenizer.
        max_sequence_length (int): Maximum sequence length.

    Returns:
        np.ndarray: Padded sequence of shape (1, max_sequence_length).
    """

    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    if not text.strip():
        raise ValueError("Input text cannot be empty.")

    # Convert text into integer token IDs
    sequences = tokenizer.texts_to_sequences([text])

    # Apply the same padding configuration used during training
    padded_sequences = pad_sequences(
        sequences,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post"
    )

    return np.asarray(padded_sequences, dtype=np.int32)
