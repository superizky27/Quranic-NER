"""
Quranic Named Entity Recognition - Data Preprocessing Module

This module handles data loading, cleaning, and preprocessing for both
BiLSTM and CRF models used in Quranic NER tasks.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QuranicNERPreprocessor:
    """
    Preprocessor for Quranic Named Entity Recognition dataset.

    Handles data loading, cleaning, vocabulary creation, and sequence
    preparation for both BiLSTM and CRF models.
    """

    def __init__(self, dataset_path: str = 'data/quranic_ner_dataset.xlsx'):
        """
        Initialize the preprocessor.

        Args:
            dataset_path (str): Path to the dataset file
        """
        self.dataset_path = dataset_path
        self.dataset = None
        self.unique_words = None
        self.unique_tags = None
        self.word_to_index = None
        self.index_to_word = None
        self.tag_to_index = None
        self.index_to_tag = None

    def load_dataset(self) -> pd.DataFrame:
        """
        Load and clean the Quranic NER dataset.

        Returns:
            pd.DataFrame: Cleaned dataset

        Raises:
            FileNotFoundError: If dataset file is not found
            Exception: For other loading errors
        """
        try:
            logger.info(f"Loading dataset from {self.dataset_path}")
            self.dataset = pd.read_excel(self.dataset_path)

            # Clean dataset - drop unnecessary columns
            columns_to_drop = ['Juz', 'no_ayat', 'pos_list', 'joint_tag', 'sentence_idx', 'Ayat']
            existing_columns = [col for col in columns_to_drop if col in self.dataset.columns]
            self.dataset = self.dataset.drop(existing_columns, axis=1)

            # Rename columns to match expected format
            self.dataset = self.dataset.rename(columns={
                'word': 'words',
                'tag': 'tags'  # level_2 is the NER tag
            })

            logger.info(f"Dataset loaded successfully. Shape: {self.dataset.shape}")
            logger.info(f"Columns: {self.dataset.columns.tolist()}")
            return self.dataset

        except FileNotFoundError:
            logger.error(f"Dataset file not found: {self.dataset_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def create_vocabulary(self) -> None:
        """
        Create word and tag vocabularies from the dataset.

        Creates mappings between words/tags and numerical indices,
        including special tokens for padding and unknown words.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        logger.info("Creating vocabulary...")

        # Extract unique words and tags
        self.unique_words = list(set(self.dataset['words'].values))
        self.unique_tags = list(set(self.dataset['tags'].values))

        # Create word-to-index mapping (reserve 0 for PAD, 1 for UNK)
        self.word_to_index = {word: idx + 2 for idx, word in enumerate(self.unique_words)}
        self.word_to_index['UNK'] = 1  # Unknown word token
        self.word_to_index['PAD'] = 0  # Padding token

        # Create reverse mapping
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}

        # Create tag-to-index mapping (reserve 0 for PAD)
        self.tag_to_index = {tag: idx + 1 for idx, tag in enumerate(self.unique_tags)}
        self.tag_to_index['PAD'] = 0  # Padding token

        # Create reverse mapping
        self.index_to_tag = {idx: tag for tag, idx in self.tag_to_index.items()}

        logger.info(f"Vocabulary created: {len(self.word_to_index)} words, {len(self.tag_to_index)} tags")

    def split_dataset(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into training and testing sets.

        Args:
            test_size (float): Proportion of dataset to use for testing
            random_state (int): Random seed for reproducibility

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and testing datasets
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        logger.info(f"Splitting dataset with test_size={test_size}")

        training_data, testing_data = train_test_split(
            self.dataset,
            test_size=test_size,
            random_state=random_state
        )

        logger.info(f"Dataset split: {len(training_data)} train, {len(testing_data)} test samples")
        return training_data, testing_data

    def prepare_sequences_for_bilstm(self, data: pd.DataFrame, max_length: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for BiLSTM model training.

        Args:
            data (pd.DataFrame): Input dataframe with 'words' and 'tags' columns
            max_length (int): Maximum sequence length

        Returns:
            Tuple[np.ndarray, np.ndarray]: Padded word sequences and one-hot encoded tags

        Raises:
            ImportError: If TensorFlow/Keras is not available
        """
        try:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            from tensorflow.keras.utils import to_categorical
        except ImportError as e:
            raise ImportError(f"TensorFlow/Keras required for BiLSTM preprocessing: {e}")

        if self.word_to_index is None or self.tag_to_index is None:
            raise ValueError("Vocabulary not created. Call create_vocabulary() first.")

        logger.info(f"Preparing BiLSTM sequences for {len(data)} samples")

        # Extract words and tags
        words_list = data['words'].values.tolist()
        tags_list = data['tags'].values.tolist()

        # Convert to index sequences
        word_sequences = [
            [self.word_to_index.get(w, self.word_to_index['UNK']) for w in words_list[i:i+max_length]]
            for i in range(0, len(words_list), max_length)
        ]

        tag_sequences = [
            [self.tag_to_index.get(t, self.tag_to_index['PAD']) for t in tags_list[i:i+max_length]]
            for i in range(0, len(tags_list), max_length)
        ]

        # Pad sequences
        padded_words = pad_sequences(word_sequences, maxlen=max_length, padding="post", value=self.word_to_index['PAD'])
        padded_tags = pad_sequences(tag_sequences, maxlen=max_length, padding="post", value=self.tag_to_index['PAD'])

        # Convert tags to one-hot encoding
        onehot_tags = [to_categorical(seq, num_classes=len(self.tag_to_index)) for seq in padded_tags]

        logger.info(f"BiLSTM sequences prepared: {padded_words.shape[0]} sequences of length {max_length}")
        return padded_words, np.array(onehot_tags)

    def prepare_sequences_for_crf(self, data: pd.DataFrame, chunk_size: int = 50) -> List[List[Tuple[str, str, str]]]:
        """
        Prepare sequences for CRF model training.

        Args:
            data (pd.DataFrame): Input dataframe
            chunk_size (int): Size of each sentence chunk

        Returns:
            List[List[Tuple[str, str, str]]]: List of sentences, each containing word-pos-tag tuples
        """
        logger.info(f"Preparing CRF sequences for {len(data)} samples")

        # Convert to list format
        data_list = data.values.tolist()

        # Create sentence chunks
        sentences = [data_list[i:i+chunk_size] for i in range(0, len(data_list), chunk_size)]

        logger.info(f"CRF sequences prepared: {len(sentences)} sentences")
        return sentences

    def get_vocabulary_info(self) -> Dict[str, Any]:
        """
        Get information about the created vocabulary.

        Returns:
            Dict[str, Any]: Vocabulary statistics and mappings
        """
        if self.word_to_index is None:
            raise ValueError("Vocabulary not created. Call create_vocabulary() first.")

        return {
            'num_words': len(self.word_to_index),
            'num_tags': len(self.tag_to_index),
            'word_to_index': self.word_to_index,
            'tag_to_index': self.tag_to_index,
            'unique_words_sample': self.unique_words[:10] if self.unique_words else [],
            'unique_tags': self.unique_tags if self.unique_tags else []
        }


def create_data_splits(dataset_path: str = 'data/quranic_ner_dataset.xlsx',
                      test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load and split dataset in one call.

    Args:
        dataset_path (str): Path to dataset file
        test_size (float): Proportion for testing

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and testing datasets
    """
    preprocessor = QuranicNERPreprocessor(dataset_path)
    preprocessor.load_dataset()
    return preprocessor.split_dataset(test_size=test_size)


if __name__ == "__main__":
    # Example usage
    preprocessor = QuranicNERPreprocessor()
    dataset = preprocessor.load_dataset()
    preprocessor.create_vocabulary()
    train_data, test_data = preprocessor.split_dataset()

    print("Preprocessing module ready!")
    print(f"Dataset shape: {dataset.shape}")
    print(f"Vocabulary: {len(preprocessor.word_to_index)} words, {len(preprocessor.tag_to_index)} tags")