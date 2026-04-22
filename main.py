#!/usr/bin/env python3
"""
Quranic Named Entity Recognition - Main Application

This is the main entry point for running Quranic NER models.
Supports both BiLSTM (deep learning) and CRF (classical ML) approaches.

Usage:
    python main.py --help
    python main.py --model bilstm --train
    python main.py --model crf --train
    python main.py --model bilstm --predict "your quranic text here"
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

# Configure logging immediately
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.preprocessing import QuranicNERPreprocessor
    from src.bilstm import QuranicBiLSTM, create_bilstm_model
    from src.crf import QuranicCRF, create_crf_model
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    logger.warning(f"TensorFlow/BiLSTM modules not available: {e}")
    logger.info("CRF functionality will still work")
    try:
        from src.preprocessing import QuranicNERPreprocessor
        from src.crf import QuranicCRF, create_crf_model
        TENSORFLOW_AVAILABLE = False
    except ImportError as e:
        logger.error(f"Critical modules not available: {e}")
        logger.error("Please install required dependencies: pip install -r requirements.txt")
        sys.exit(1)


class QuranicNERApplication:
    """
    Main application class for Quranic Named Entity Recognition.

    Provides a unified interface for training and using both BiLSTM and CRF models.
    """

    def __init__(self):
        """Initialize the application."""
        self.preprocessor = None
        self.bilstm_model = None
        self.crf_model = None
        self.is_trained = {'bilstm': False, 'crf': False}
        self.tensorflow_available = TENSORFLOW_AVAILABLE

        if not self.tensorflow_available:
            logger.warning("TensorFlow not available. BiLSTM functionality disabled.")

        logger.info("Quranic NER Application initialized")

    def setup_preprocessing(self) -> None:
        """
        Setup the data preprocessor.

        Loads dataset and creates vocabulary mappings.
        """
        logger.info("Setting up data preprocessing...")

        self.preprocessor = QuranicNERPreprocessor()
        self.preprocessor.load_dataset()
        self.preprocessor.create_vocabulary()

        vocab_info = self.preprocessor.get_vocabulary_info()
        logger.info(f"Preprocessing setup complete: {vocab_info['num_words']} words, {vocab_info['num_tags']} tags")

    def train_bilstm(self, epochs: int = 25, save_model: bool = True) -> dict:
        """
        Train the BiLSTM model.

        Args:
            epochs (int): Number of training epochs
            save_model (bool): Whether to save the trained model

        Returns:
            dict: Training results and metrics

        Raises:
            RuntimeError: If TensorFlow is not available
        """
        if not self.tensorflow_available:
            raise RuntimeError("TensorFlow not available. Cannot train BiLSTM model.")

        if self.preprocessor is None:
            self.setup_preprocessing()

        logger.info("🚀 Training BiLSTM model...")

        # Prepare training data
        train_data, test_data = self.preprocessor.split_dataset()
        x_train, y_train = self.preprocessor.prepare_sequences_for_bilstm(train_data)
        x_test, y_test = self.preprocessor.prepare_sequences_for_bilstm(test_data)

        # Create and train model
        self.bilstm_model = create_bilstm_model(self.preprocessor)
        training_history = self.bilstm_model.train(x_train, y_train, epochs=epochs)

        # Evaluate model
        test_metrics = self.bilstm_model.evaluate(x_test, y_test)

        # Save model if requested
        if save_model:
            model_path = Path("models/bilstm_model.h5")
            model_path.parent.mkdir(exist_ok=True)
            self.bilstm_model.save_model(str(model_path))
            logger.info(f"BiLSTM model saved to {model_path}")

        self.is_trained['bilstm'] = True

        results = {
            'model_type': 'BiLSTM',
            'training_history': training_history,
            'test_metrics': test_metrics,
            'epochs': epochs
        }

        logger.info("✅ BiLSTM training completed!")
        logger.info(f"Test Results: F1={test_metrics['f1_score']:.4f}, Accuracy={test_metrics['accuracy']:.4f}")

        return results

    def train_crf(self, save_model: bool = True) -> dict:
        """
        Train the CRF model.

        Args:
            save_model (bool): Whether to save the trained model

        Returns:
            dict: Training results and metrics
        """
        if self.preprocessor is None:
            self.setup_preprocessing()

        logger.info("🚀 Training CRF model...")

        # Prepare training data
        train_data, test_data = self.preprocessor.split_dataset()
        train_sentences = self.preprocessor.prepare_sequences_for_crf(train_data)
        test_sentences = self.preprocessor.prepare_sequences_for_crf(test_data)

        # Prepare features and labels
        crf_trainer = QuranicCRF()
        crf_trainer.build_model()

        x_train, y_train = crf_trainer.prepare_training_data(train_sentences)
        x_test, y_test = crf_trainer.prepare_training_data(test_sentences)

        # Train model
        self.crf_model = crf_trainer
        self.crf_model.train(x_train, y_train)

        # Evaluate model
        y_pred = self.crf_model.predict(x_test)
        test_metrics = self.crf_model.evaluate(y_test, y_pred)

        # Save model if requested
        if save_model:
            model_path = Path("models/crf_model.pkl")
            model_path.parent.mkdir(exist_ok=True)
            self.crf_model.save_model(str(model_path))
            logger.info(f"CRF model saved to {model_path}")

        self.is_trained['crf'] = True

        results = {
            'model_type': 'CRF',
            'test_metrics': test_metrics,
            'model_info': self.crf_model.get_model_info()
        }

        logger.info("✅ CRF training completed!")
        logger.info(f"Test Results: Weighted F1={test_metrics['weighted_f1']:.4f}")

        return results

    def predict_with_bilstm(self, text: str) -> list:
        """
        Make predictions using the BiLSTM model.

        Args:
            text (str): Input text to analyze

        Returns:
            list: List of (word, predicted_tag) tuples

        Raises:
            ValueError: If BiLSTM model is not trained
            RuntimeError: If TensorFlow is not available
        """
        if not self.tensorflow_available:
            raise RuntimeError("TensorFlow not available. Cannot use BiLSTM model.")

        if not self.is_trained['bilstm'] or self.bilstm_model is None:
            raise ValueError("BiLSTM model not trained. Run train_bilstm() first.")

        logger.info(f"🔍 Predicting with BiLSTM: '{text[:50]}...'")
        predictions = self.bilstm_model.predict_text(text, self.preprocessor)

        return predictions

    def predict_with_crf(self, text: str) -> list:
        """
        Make predictions using the CRF model.

        Args:
            text (str): Input text to analyze

        Returns:
            list: List of (word, predicted_tag) tuples

        Raises:
            ValueError: If CRF model is not trained
        """
        if not self.is_trained['crf'] or self.crf_model is None:
            raise ValueError("CRF model not trained. Run train_crf() first.")

        logger.info(f"🔍 Predicting with CRF: '{text[:50]}...'")
        predictions = self.crf_model.predict_text(text)

        return predictions

    def load_models(self) -> None:
        """
        Load pre-trained models from disk.
        """
        bilstm_path = Path("models/bilstm_model.h5")
        crf_path = Path("models/crf_model.pkl")

        if bilstm_path.exists():
            if self.bilstm_model is None:
                vocab_info = self.preprocessor.get_vocabulary_info() if self.preprocessor else None
                if vocab_info:
                    self.bilstm_model = QuranicBiLSTM(
                        vocab_size=len(vocab_info['word_to_index']),
                        num_tags=len(vocab_info['tag_to_index'])
                    )
            if self.bilstm_model:
                self.bilstm_model.load_model(str(bilstm_path))
                self.is_trained['bilstm'] = True
                logger.info("BiLSTM model loaded from disk")

        if crf_path.exists():
            if self.crf_model is None:
                self.crf_model = QuranicCRF()
            self.crf_model.load_model(str(crf_path))
            self.is_trained['crf'] = True
            logger.info("CRF model loaded from disk")

    def compare_models(self, test_text: Optional[str] = None) -> dict:
        """
        Compare performance of both models.

        Args:
            test_text (str, optional): Text to test predictions on

        Returns:
            dict: Comparison results
        """
        if self.preprocessor is None:
            self.setup_preprocessing()

        results = {
            'bilstm': {},
            'crf': {},
            'comparison': {}
        }

        # Get test data for evaluation
        _, test_data = self.preprocessor.split_dataset()

        # Evaluate BiLSTM
        if self.is_trained['bilstm']:
            x_test, y_test = self.preprocessor.prepare_sequences_for_bilstm(test_data)
            bilstm_metrics = self.bilstm_model.evaluate(x_test, y_test)
            results['bilstm']['metrics'] = bilstm_metrics

        # Evaluate CRF
        if self.is_trained['crf']:
            test_sentences = self.preprocessor.prepare_sequences_for_crf(test_data)
            x_test, y_test = self.crf_model.prepare_training_data(test_sentences)
            y_pred = self.crf_model.predict(x_test)
            crf_metrics = self.crf_model.evaluate(y_test, y_pred)
            results['crf']['metrics'] = crf_metrics

        # Test predictions on sample text
        if test_text:
            if self.is_trained['bilstm']:
                results['bilstm']['sample_prediction'] = self.predict_with_bilstm(test_text)
            if self.is_trained['crf']:
                results['crf']['sample_prediction'] = self.predict_with_crf(test_text)

        # Create comparison
        if results['bilstm'].get('metrics') and results['crf'].get('metrics'):
            bilstm_f1 = results['bilstm']['metrics'].get('f1_score', 0)
            crf_f1 = results['crf']['metrics'].get('weighted_f1', 0)

            results['comparison'] = {
                'bilstm_f1': bilstm_f1,
                'crf_f1': crf_f1,
                'winner': 'BiLSTM' if bilstm_f1 > crf_f1 else 'CRF',
                'difference': abs(bilstm_f1 - crf_f1)
            }

        return results


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Quranic Named Entity Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --model bilstm --train
  python main.py --model crf --train --save-model
  python main.py --model bilstm --predict "your quranic text here"
  python main.py --compare --text "sample text"
        """
    )

    parser.add_argument(
        '--model', '-m',
        choices=['bilstm', 'crf', 'both'],
        default='both',
        help='Model to use (default: both)'
    )

    parser.add_argument(
        '--train', '-t',
        action='store_true',
        help='Train the specified model(s)'
    )

    parser.add_argument(
        '--predict', '-p',
        type=str,
        help='Predict NER tags for the given text'
    )

    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=25,
        help='Number of training epochs for BiLSTM (default: 25)'
    )

    parser.add_argument(
        '--save-model', '-s',
        action='store_true',
        help='Save trained model(s) to disk'
    )

    parser.add_argument(
        '--compare', '-c',
        action='store_true',
        help='Compare both models performance'
    )

    parser.add_argument(
        '--text',
        type=str,
        help='Text to use for prediction or comparison'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize application
    app = QuranicNERApplication()

    try:
        # Setup preprocessing
        app.setup_preprocessing()

        # Load existing models if available
        app.load_models()

        # Train models
        if args.train:
            if args.model in ['bilstm', 'both']:
                if not app.tensorflow_available:
                    logger.error("Cannot train BiLSTM model: TensorFlow not available")
                    logger.info("Please install TensorFlow: pip install tensorflow")
                else:
                    bilstm_results = app.train_bilstm(epochs=args.epochs, save_model=args.save_model)
                    print(f"\nBiLSTM Results: F1 = {bilstm_results['test_metrics']['f1_score']:.4f}")

            if args.model in ['crf', 'both']:
                crf_results = app.train_crf(save_model=args.save_model)
                print(f"\nCRF Results: Weighted F1 = {crf_results['test_metrics']['weighted_f1']:.4f}")

        # Make predictions
        if args.predict:
            text = args.predict

            if args.model in ['bilstm', 'both'] and app.tensorflow_available and app.is_trained['bilstm']:
                print("\n🔍 BiLSTM Predictions:")
                bilstm_preds = app.predict_with_bilstm(text)
                for word, tag in bilstm_preds:
                    print(f"  {word:<15} → {tag}")

            if args.model in ['crf', 'both'] and app.is_trained['crf']:
                print("\n🔍 CRF Predictions:")
                crf_preds = app.predict_with_crf(text)
                for word, tag in crf_preds:
                    print(f"  {word:<15} → {tag}")

            if args.model in ['bilstm', 'both'] and not app.tensorflow_available:
                logger.warning("BiLSTM predictions skipped: TensorFlow not available")

        # Compare models
        if args.compare:
            test_text = args.text or "يَا أَيُّهَا الَّذِينَ آمَنُوا اتَّقُوا اللَّهَ وَكُونُوا مَعَ الصَّادِقِينَ"
            comparison = app.compare_models(test_text)

            print("\n" + "="*60)
            print("📊 MODEL COMPARISON")
            print("="*60)

            if 'bilstm' in comparison and 'metrics' in comparison['bilstm']:
                print(f"BiLSTM F1 Score: {comparison['bilstm']['metrics']['f1_score']:.4f}")

            if 'crf' in comparison and 'metrics' in comparison['crf']:
                print(f"CRF F1 Score:    {comparison['crf']['metrics']['weighted_f1']:.4f}")

            if 'comparison' in comparison:
                comp = comparison['comparison']
                print(f"Winner: {comp['winner']}")
                print(f"Difference: {comp['difference']:.4f}")

            if test_text and ('bilstm' in comparison or 'crf' in comparison):
                print(f"\nSample predictions on: '{test_text[:30]}...'")
                if 'sample_prediction' in comparison.get('bilstm', {}) and app.tensorflow_available:
                    print("BiLSTM:", comparison['bilstm']['sample_prediction'][:3])
                if 'sample_prediction' in comparison.get('crf', {}):
                    print("CRF:   ", comparison['crf']['sample_prediction'][:3])

        # If no action specified, show help
        if not any([args.train, args.predict, args.compare]):
            parser.print_help()

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()