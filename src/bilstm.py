"""
Quranic Named Entity Recognition - BiLSTM Model Module

This module implements the Bidirectional Long Short-Term Memory (BiLSTM)
model for Named Entity Recognition on Quranic text.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, TimeDistributed, Dense, SpatialDropout1D
from tensorflow.keras import backend as K
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import logging
from typing import List, Tuple, Dict, Any, Optional
import seaborn as sns
import logging
from typing import Tuple, Dict, Any, Optional

from .preprocessing import QuranicNERPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QuranicBiLSTM:
    """
    Bidirectional LSTM model for Quranic Named Entity Recognition.

    This class implements a deep learning approach using BiLSTM architecture
    with custom metrics for NER task evaluation.
    """

    def __init__(self, vocab_size: int, num_tags: int, max_length: int = 50,
                 embedding_dim: int = 50, lstm_units: int = 100):
        """
        Initialize the BiLSTM model.

        Args:
            vocab_size (int): Size of word vocabulary
            num_tags (int): Number of NER tags
            max_length (int): Maximum sequence length
            embedding_dim (int): Word embedding dimension
            lstm_units (int): Number of LSTM units
        """
        self.vocab_size = vocab_size
        self.num_tags = num_tags
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None
        self.history = None

        logger.info(f"BiLSTM model initialized: vocab={vocab_size}, tags={num_tags}, max_len={max_length}")

    def build_model(self) -> Model:
        """
        Build the BiLSTM neural network architecture.

        Architecture:
        Input → Embedding → SpatialDropout1D → BiLSTM → TimeDistributed(Dense) → Output

        Returns:
            Model: Compiled Keras model
        """
        logger.info("Building BiLSTM model architecture...")

        # Input layer
        input_layer = Input(shape=(self.max_length,), name='input_words')

        # Embedding layer
        embedding_layer = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_length,
            name='word_embeddings'
        )(input_layer)

        # Spatial dropout for regularization
        dropout_layer = SpatialDropout1D(rate=0.2, name='spatial_dropout')(embedding_layer)

        # Bidirectional LSTM
        bilstm_layer = Bidirectional(
            LSTM(units=self.lstm_units, return_sequences=True, recurrent_dropout=0.1),
            name='bidirectional_lstm'
        )(dropout_layer)

        # TimeDistributed Dense layer for sequence prediction
        output_layer = TimeDistributed(
            Dense(self.num_tags, activation='softmax'),
            name='ner_predictions'
        )(bilstm_layer)

        # Create model
        self.model = Model(input_layer, output_layer, name='Quranic_BiLSTM_NER')

        # Compile with custom metrics
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', self.f1_metric, self.precision_metric, self.recall_metric]
        )

        logger.info("BiLSTM model built and compiled successfully")
        return self.model

    @staticmethod
    def recall_metric(y_true, y_pred) -> float:
        """
        Calculate recall metric.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities

        Returns:
            float: Recall score
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    @staticmethod
    def precision_metric(y_true, y_pred) -> float:
        """
        Calculate precision metric.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities

        Returns:
            float: Precision score
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    @staticmethod
    def f1_metric(y_true, y_pred) -> float:
        """
        Calculate F1 score (harmonic mean of precision and recall).

        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities

        Returns:
            float: F1 score
        """
        precision = QuranicBiLSTM.precision_metric(y_true, y_pred)
        recall = QuranicBiLSTM.recall_metric(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def train(self, x_train: np.ndarray, y_train: np.ndarray,
              x_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 25, batch_size: int = 128, validation_split: float = 0.1) -> Dict[str, Any]:
        """
        Train the BiLSTM model.

        Args:
            x_train (np.ndarray): Training input sequences
            y_train (np.ndarray): Training target sequences
            x_val (np.ndarray, optional): Validation input sequences
            y_val (np.ndarray, optional): Validation target sequences
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of training data for validation

        Returns:
            Dict[str, Any]: Training history and metrics
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        logger.info(f"Training BiLSTM model for {epochs} epochs...")

        # Prepare validation data
        validation_data = None
        if x_val is not None and y_val is not None:
            validation_data = (x_val, y_val)
            validation_split = 0.0

        # Train model
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            validation_data=validation_data,
            verbose=1
        )

        logger.info("BiLSTM model training completed")
        return self.history.history

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray, batch_size: int = 128) -> Dict[str, float]:
        """
        Evaluate the trained model on test data.

        Args:
            x_test (np.ndarray): Test input sequences
            y_test (np.ndarray): Test target sequences
            batch_size (int): Batch size for evaluation

        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        logger.info("Evaluating BiLSTM model on test data...")

        # Evaluate model
        evaluation_results = self.model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)

        # Create metrics dictionary
        metrics = {
            'loss': float(evaluation_results[0]),
            'accuracy': float(evaluation_results[1]),
            'f1_score': float(evaluation_results[2]),
            'precision': float(evaluation_results[3]),
            'recall': float(evaluation_results[4])
        }

        logger.info(f"Evaluation completed: F1={metrics['f1_score']:.4f}, Accuracy={metrics['accuracy']:.4f}")
        return metrics

    def predict(self, input_sequences: np.ndarray) -> np.ndarray:
        """
        Make predictions on input sequences.

        Args:
            input_sequences (np.ndarray): Input sequences to predict

        Returns:
            np.ndarray: Predicted tag indices
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        logger.info(f"Making predictions on {len(input_sequences)} sequences")

        predictions = self.model.predict(input_sequences, verbose=0)
        predicted_indices = np.argmax(predictions, axis=-1)

        return predicted_indices

    def predict_text(self, text: str, preprocessor: QuranicNERPreprocessor) -> List[Tuple[str, str]]:
        """
        Predict NER tags for a given text.

        Args:
            text (str): Input text to analyze
            preprocessor (QuranicNERPreprocessor): Preprocessor with vocabulary mappings

        Returns:
            List[Tuple[str, str]]: List of (word, predicted_tag) tuples

        Raises:
            ImportError: If TensorFlow/Keras is not available
        """
        try:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
        except ImportError as e:
            raise ImportError(f"TensorFlow/Keras required for text prediction: {e}")

        # Tokenize text
        words = text.split()

        # Truncate if too long
        if len(words) > self.max_length:
            logger.warning(f"Text has {len(words)} words, truncating to {self.max_length}")
            words = words[:self.max_length]

        # Convert to indices
        word_indices = [
            preprocessor.word_to_index.get(word, preprocessor.word_to_index['UNK'])
            for word in words
        ]

        # Pad sequence
        padded_sequence = pad_sequences([word_indices], maxlen=self.max_length,
                                      padding="post", value=preprocessor.word_to_index['PAD'])

        # Make prediction
        predictions = self.predict(padded_sequence)

        # Convert back to tags
        predicted_tags = [
            (words[idx], preprocessor.index_to_tag[predictions[0][idx]])
            for idx in range(len(words))
        ]

        return predicted_tags

    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training history (loss and metrics).

        Args:
            save_path (str, optional): Path to save the plot
        """
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()

        # Plot accuracy
        axes[0, 1].plot(self.history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in self.history.history:
            axes[0, 1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()

        # Plot F1 score
        if 'f1_metric' in self.history.history:
            axes[1, 0].plot(self.history.history['f1_metric'], label='Training F1')
            if 'val_f1_metric' in self.history.history:
                axes[1, 0].plot(self.history.history['val_f1_metric'], label='Validation F1')
            axes[1, 0].set_title('F1 Score')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].legend()

        # Plot precision and recall
        if 'precision_metric' in self.history.history and 'recall_metric' in self.history.history:
            axes[1, 1].plot(self.history.history['precision_metric'], label='Training Precision')
            axes[1, 1].plot(self.history.history['recall_metric'], label='Training Recall')
            if 'val_precision_metric' in self.history.history:
                axes[1, 1].plot(self.history.history['val_precision_metric'], label='Validation Precision')
            if 'val_recall_metric' in self.history.history:
                axes[1, 1].plot(self.history.history['val_recall_metric'], label='Validation Recall')
            axes[1, 1].set_title('Precision & Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")

        plt.show()

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.

        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Build and train the model first.")

        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.

        Args:
            filepath (str): Path to the saved model
        """
        self.model = tf.keras.models.load_model(
            filepath,
            custom_objects={
                'f1_metric': self.f1_metric,
                'precision_metric': self.precision_metric,
                'recall_metric': self.recall_metric
            }
        )
        logger.info(f"Model loaded from {filepath}")


def create_bilstm_model(preprocessor: QuranicNERPreprocessor,
                       embedding_dim: int = 50,
                       lstm_units: int = 100) -> QuranicBiLSTM:
    """
    Convenience function to create a BiLSTM model with preprocessor.

    Args:
        preprocessor (QuranicNERPreprocessor): Initialized preprocessor
        embedding_dim (int): Word embedding dimension
        lstm_units (int): Number of LSTM units

    Returns:
        QuranicBiLSTM: Configured BiLSTM model
    """
    vocab_info = preprocessor.get_vocabulary_info()

    model = QuranicBiLSTM(
        vocab_size=len(vocab_info['word_to_index']),
        num_tags=len(vocab_info['tag_to_index']),
        embedding_dim=embedding_dim,
        lstm_units=lstm_units
    )

    model.build_model()
    return model


if __name__ == "__main__":
    # Example usage
    from preprocessing import QuranicNERPreprocessor

    # Initialize preprocessor
    preprocessor = QuranicNERPreprocessor()
    preprocessor.load_dataset()
    preprocessor.create_vocabulary()

    # Create BiLSTM model
    bilstm_model = create_bilstm_model(preprocessor)

    print("BiLSTM module ready!")
    print(f"Model summary:")
    bilstm_model.model.summary()