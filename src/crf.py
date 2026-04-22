"""
Quranic Named Entity Recognition - CRF Model Module

This module implements the Conditional Random Field (CRF) model
for Named Entity Recognition on Quranic text.
"""

import sklearn_crfsuite
from sklearn_crfsuite import scorers, metrics
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Any, Optional

from .preprocessing import QuranicNERPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QuranicCRF:
    """
    Conditional Random Field model for Quranic Named Entity Recognition.

    This class implements a classical machine learning approach using CRF
    with extensive feature engineering for sequence labeling.
    """

    def __init__(self, algorithm: str = 'lbfgs', c1: float = 0.1, c2: float = 0.1,
                 max_iterations: int = 100):
        """
        Initialize the CRF model.

        Args:
            algorithm (str): Optimization algorithm ('lbfgs', 'l2sgd', etc.)
            c1 (float): L1 regularization coefficient
            c2 (float): L2 regularization coefficient
            max_iterations (int): Maximum iterations for convergence
        """
        self.algorithm = algorithm
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self.model = None
        self.feature_functions = []

        logger.info(f"CRF model initialized: algorithm={algorithm}, c1={c1}, c2={c2}")

    def build_model(self) -> sklearn_crfsuite.CRF:
        """
        Build the CRF model with specified parameters.

        Returns:
            sklearn_crfsuite.CRF: Configured CRF model
        """
        logger.info("Building CRF model...")

        self.model = sklearn_crfsuite.CRF(
            algorithm=self.algorithm,
            c1=self.c1,
            c2=self.c2,
            max_iterations=self.max_iterations,
            all_possible_transitions=True,
            verbose=True
        )

        logger.info("CRF model built successfully")
        return self.model

    def extract_word_features(self, sentence_tokens: List[Tuple[str, str, str]],
                            token_position: int) -> Dict[str, Any]:
        """
        Extract comprehensive features for a word at given position.

        Features include:
        - Word properties (lowercase, suffixes, case)
        - POS tag information
        - Context features (previous/next word)
        - Boundary markers

        Args:
            sentence_tokens (List[Tuple[str, str, str]]): List of (word, pos_tag, ner_tag) tuples
            token_position (int): Position of token in sentence

        Returns:
            Dict[str, Any]: Feature dictionary for the token

        Raises:
            IndexError: If token_position is out of range
        """
        try:
            if token_position >= len(sentence_tokens):
                raise IndexError(f"Token position {token_position} out of range")

            # Extract current word and POS tag
            current_word, current_pos, _ = sentence_tokens[token_position]

            # Initialize feature dictionary
            features = {
                'bias': 1.0,
                'word_lowercase': current_word.lower(),
                'word_suffix_3chars': current_word[-3:],
                'word_suffix_2chars': current_word[-2:],
                'word_is_uppercase': current_word.isupper(),
                'word_is_titlecase': current_word.istitle(),
                'word_is_digit': current_word.isdigit(),
                'word_length': len(current_word),
                'pos_tag_full': current_pos,
                'pos_tag_first_2chars': current_pos[:2],
                'pos_tag_last_2chars': current_pos[-2:] if len(current_pos) >= 2 else current_pos,
            }

            # Add features for previous word (if exists)
            if token_position > 0:
                prev_word, prev_pos, _ = sentence_tokens[token_position - 1]
                features.update({
                    'prev_word_lowercase': prev_word.lower(),
                    'prev_word_is_titlecase': prev_word.istitle(),
                    'prev_word_is_uppercase': prev_word.isupper(),
                    'prev_word_length': len(prev_word),
                    'prev_pos_tag_full': prev_pos,
                    'prev_pos_tag_first_2chars': prev_pos[:2],
                    'prev_pos_tag_last_2chars': prev_pos[-2:] if len(prev_pos) >= 2 else prev_pos,
                    # Transition features
                    'pos_transition': f"{prev_pos}→{current_pos}",
                    'word_pos_pair': f"{prev_word.lower()}_{current_pos}",
                })
            else:
                # Beginning of sentence marker
                features['is_begin_of_sentence'] = True

            # Add features for next word (if exists)
            if token_position < len(sentence_tokens) - 1:
                next_word, next_pos, _ = sentence_tokens[token_position + 1]
                features.update({
                    'next_word_lowercase': next_word.lower(),
                    'next_word_is_titlecase': next_word.istitle(),
                    'next_word_is_uppercase': next_word.isupper(),
                    'next_word_length': len(next_word),
                    'next_pos_tag_full': next_pos,
                    'next_pos_tag_first_2chars': next_pos[:2],
                    'next_pos_tag_last_2chars': next_pos[-2:] if len(next_pos) >= 2 else next_pos,
                    # Look-ahead transition
                    'next_pos_transition': f"{current_pos}→{next_pos}",
                    'word_next_pos_pair': f"{current_word.lower()}_{next_pos}",
                })
            else:
                # End of sentence marker
                features['is_end_of_sentence'] = True

            # Add position features
            features.update({
                'position_in_sentence': token_position,
                'relative_position': token_position / len(sentence_tokens),
                'is_first_quarter': token_position < len(sentence_tokens) / 4,
                'is_last_quarter': token_position >= 3 * len(sentence_tokens) / 4,
            })

            return features

        except IndexError as e:
            logger.error(f"Index error in feature extraction: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error extracting word features: {str(e)}")
            raise

    def extract_sentence_features(self, sentence_tokens: List[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
        """
        Extract features for all words in a sentence.

        Args:
            sentence_tokens (List[Tuple[str, str, str]]): List of (word, pos_tag, ner_tag) tuples

        Returns:
            List[Dict[str, Any]]: List of feature dictionaries
        """
        try:
            return [
                self.extract_word_features(sentence_tokens, token_idx)
                for token_idx in range(len(sentence_tokens))
            ]
        except Exception as e:
            logger.error(f"Error extracting sentence features: {str(e)}")
            raise

    def extract_sentence_labels(self, sentence_tokens: List[Tuple[str, str, str]]) -> List[str]:
        """
        Extract NER labels for all words in a sentence.

        Args:
            sentence_tokens (List[Tuple[str, str, str]]): List of (word, pos_tag, ner_tag) tuples

        Returns:
            List[str]: List of NER labels
        """
        try:
            return [ner_tag for _, _, ner_tag in sentence_tokens]
        except Exception as e:
            logger.error(f"Error extracting sentence labels: {str(e)}")
            raise

    def extract_sentence_tokens(self, sentence_tokens: List[Tuple[str, str, str]]) -> List[str]:
        """
        Extract words (tokens) from a sentence.

        Args:
            sentence_tokens (List[Tuple[str, str, str]]): List of (word, pos_tag, ner_tag) tuples

        Returns:
            List[str]: List of words
        """
        try:
            return [word for word, _, _ in sentence_tokens]
        except Exception as e:
            logger.error(f"Error extracting sentence tokens: {str(e)}")
            raise

    def prepare_training_data(self, sentences: List[List[Tuple[str, str, str]]]) -> Tuple[List[List[Dict]], List[List[str]]]:
        """
        Prepare training data by extracting features and labels.

        Args:
            sentences (List[List[Tuple[str, str, str]]]): List of sentences

        Returns:
            Tuple[List[List[Dict]], List[List[str]]]: Features and labels for training
        """
        logger.info(f"Preparing training data from {len(sentences)} sentences...")

        x_features = []
        y_labels = []

        for sentence in sentences:
            if sentence:  # Skip empty sentences
                x_features.append(self.extract_sentence_features(sentence))
                y_labels.append(self.extract_sentence_labels(sentence))

        logger.info(f"Training data prepared: {len(x_features)} sentences")
        return x_features, y_labels

    def train(self, x_train: List[List[Dict]], y_train: List[List[str]]) -> sklearn_crfsuite.CRF:
        """
        Train the CRF model.

        Args:
            x_train (List[List[Dict]]): Training features
            y_train (List[List[str]]): Training labels

        Returns:
            sklearn_crfsuite.CRF: Trained CRF model

        Raises:
            ValueError: If training data is invalid
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        if not x_train or not y_train:
            raise ValueError("Training data cannot be empty")

        logger.info("Training CRF model...")
        logger.info(f"Training on {len(x_train)} sentences with {len(x_train[0][0])} features per token")

        try:
            self.model.fit(x_train, y_train)
            logger.info("CRF model training completed successfully")
            return self.model
        except Exception as e:
            logger.error(f"Error during CRF training: {str(e)}")
            raise

    def predict(self, x_test: List[List[Dict]]) -> List[List[str]]:
        """
        Make predictions on test data.

        Args:
            x_test (List[List[Dict]]): Test features

        Returns:
            List[List[str]]: Predicted labels

        Raises:
            ValueError: If model not trained
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        logger.info(f"Making predictions on {len(x_test)} test sentences")

        try:
            predictions = self.model.predict(x_test)
            logger.info("Predictions completed")
            return predictions
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def predict_text(self, text: str) -> List[Tuple[str, str]]:
        """
        Predict NER tags for a given text.

        Args:
            text (str): Input text to analyze

        Returns:
            List[Tuple[str, str]]: List of (word, predicted_tag) tuples

        Raises:
            ValueError: If model not trained
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Tokenize text (basic tokenization - in production, use proper Arabic tokenizer)
        words = text.split()

        # Create dummy POS tags (since we don't have POS tagger in this simple version)
        # In production, you would use a proper POS tagger
        dummy_pos_tags = ['NN'] * len(words)  # Default to noun

        # Create sentence tokens
        sentence_tokens = [(word, pos, 'O') for word, pos in zip(words, dummy_pos_tags)]

        # Extract features
        features = self.extract_sentence_features(sentence_tokens)

        # Make prediction
        predictions = self.model.predict([features])[0]

        # Return word-tag pairs
        return list(zip(words, predictions))

    def evaluate(self, y_true: List[List[str]], y_pred: List[List[str]],
                exclude_o_tag: bool = True) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            y_true (List[List[str]]): True labels
            y_pred (List[List[str]]): Predicted labels
            exclude_o_tag (bool): Whether to exclude 'O' tag from evaluation

        Returns:
            Dict[str, float]: Evaluation metrics
        """
        logger.info("Evaluating CRF model performance...")

        # Flatten predictions and true labels
        true_labels_flat = [label for sent in y_true for label in sent]
        pred_labels_flat = [label for sent in y_pred for label in sent]

        # Get unique labels
        labels = list(set(true_labels_flat))
        if exclude_o_tag and 'O' in labels:
            labels.remove('O')

        # Calculate metrics
        weighted_f1 = metrics.flat_f1_score(y_true, y_pred, average='weighted', labels=labels)
        macro_f1 = metrics.flat_f1_score(y_true, y_pred, average='macro', labels=labels)
        micro_f1 = metrics.flat_f1_score(y_true, y_pred, average='micro', labels=labels)

        # Calculate precision and recall
        weighted_precision = metrics.flat_precision_score(y_true, y_pred, average='weighted', labels=labels)
        weighted_recall = metrics.flat_recall_score(y_true, y_pred, average='weighted', labels=labels)

        evaluation_results = {
            'weighted_f1': float(weighted_f1),
            'macro_f1': float(macro_f1),
            'micro_f1': float(micro_f1),
            'weighted_precision': float(weighted_precision),
            'weighted_recall': float(weighted_recall),
            'num_labels_evaluated': len(labels),
            'labels_evaluated': labels
        }

        logger.info(f"Evaluation completed: Weighted F1 = {weighted_f1:.4f}")
        return evaluation_results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.

        Returns:
            Dict[str, Any]: Model information
        """
        if self.model is None:
            raise ValueError("Model not built")

        # Check if model has been trained (has classes_)
        if not hasattr(self.model, 'classes_') or self.model.classes_ is None:
            raise ValueError("Model not trained")

        return {
            'algorithm': self.algorithm,
            'c1': self.c1,
            'c2': self.c2,
            'max_iterations': self.max_iterations,
            'classes': list(self.model.classes_),
            'num_classes': len(self.model.classes_),
            'transitions': len(self.model.transition_features_) if self.model.transition_features_ else 0,
            'state_features': len(self.model.state_features_) if self.model.state_features_ else 0
        }

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.

        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        import joblib
        joblib.dump(self.model, filepath)
        logger.info(f"CRF model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.

        Args:
            filepath (str): Path to the saved model
        """
        import joblib
        self.model = joblib.load(filepath)
        logger.info(f"CRF model loaded from {filepath}")


def create_crf_model(c1: float = 0.1, c2: float = 0.1,
                    max_iterations: int = 100) -> QuranicCRF:
    """
    Convenience function to create a CRF model.

    Args:
        c1 (float): L1 regularization coefficient
        c2 (float): L2 regularization coefficient
        max_iterations (int): Maximum iterations

    Returns:
        QuranicCRF: Configured CRF model
    """
    model = QuranicCRF(c1=c1, c2=c2, max_iterations=max_iterations)
    model.build_model()
    return model


if __name__ == "__main__":
    # Example usage
    print("CRF module ready!")
    print("Use this module for classical ML approach to Quranic NER")

    # Example of creating a model
    crf_model = create_crf_model()
    print(f"CRF model created with {crf_model.get_model_info()['num_classes']} classes")