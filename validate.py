#!/usr/bin/env python3
"""
Quranic NER Validation Script

This script validates that all modules can be imported and basic functionality works.
Run this to ensure the installation is correct before training models.
"""

import sys
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing module imports...")

    try:
        from src.preprocessing import QuranicNERPreprocessor
        logger.info("✅ Preprocessing module imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import preprocessing module: {e}")
        return False

    try:
        from src.crf import QuranicCRF, create_crf_model
        logger.info("✅ CRF module imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import CRF module: {e}")
        return False

    # Test BiLSTM import (may fail if TensorFlow not available)
    try:
        from src.bilstm import QuranicBiLSTM, create_bilstm_model
        logger.info("✅ BiLSTM module imported successfully")
        tensorflow_available = True
    except ImportError as e:
        logger.warning(f"⚠️  BiLSTM module not available (TensorFlow not installed): {e}")
        logger.info("CRF functionality will still work")
        tensorflow_available = False

    return True, tensorflow_available


def test_preprocessing():
    """Test preprocessing functionality."""
    logger.info("Testing preprocessing functionality...")

    try:
        from src.preprocessing import QuranicNERPreprocessor

        # Initialize preprocessor
        preprocessor = QuranicNERPreprocessor()

        # Test dataset loading
        dataset = preprocessor.load_dataset()
        logger.info(f"✅ Dataset loaded: {dataset.shape[0]} rows, {dataset.shape[1]} columns")

        # Test vocabulary creation
        preprocessor.create_vocabulary()
        vocab_info = preprocessor.get_vocabulary_info()
        logger.info(f"✅ Vocabulary created: {vocab_info['num_words']} words, {vocab_info['num_tags']} tags")

        # Test dataset splitting
        train_data, test_data = preprocessor.split_dataset()
        logger.info(f"✅ Dataset split: {len(train_data)} train, {len(test_data)} test samples")

        return True

    except Exception as e:
        logger.error(f"❌ Preprocessing test failed: {e}")
        return False


def test_crf():
    """Test CRF functionality."""
    logger.info("Testing CRF functionality...")

    try:
        from src.crf import create_crf_model

        # Create model
        crf_model = create_crf_model()
        logger.info("✅ CRF model created successfully")

        # Test model info (only if trained, otherwise just check creation)
        try:
            model_info = crf_model.get_model_info()
            logger.info(f"✅ CRF model info: {model_info['num_classes']} classes")
        except ValueError:
            logger.info("✅ CRF model created (not yet trained)")

        return True

    except Exception as e:
        logger.error(f"❌ CRF test failed: {e}")
        return False


def test_bilstm(tensorflow_available):
    """Test BiLSTM functionality if TensorFlow is available."""
    if not tensorflow_available:
        logger.info("⏭️  Skipping BiLSTM test (TensorFlow not available)")
        return True

    logger.info("Testing BiLSTM functionality...")

    try:
        from src.preprocessing import QuranicNERPreprocessor
        from src.bilstm import create_bilstm_model

        # Setup preprocessor
        preprocessor = QuranicNERPreprocessor()
        preprocessor.load_dataset()
        preprocessor.create_vocabulary()

        # Create BiLSTM model
        bilstm_model = create_bilstm_model(preprocessor)
        logger.info("✅ BiLSTM model created successfully")

        # Test model summary (without printing full summary)
        logger.info(f"✅ BiLSTM model built with {bilstm_model.vocab_size} vocab size")

        return True

    except Exception as e:
        logger.error(f"❌ BiLSTM test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("🔍 QURANIC NER VALIDATION SCRIPT")
    print("=" * 60)

    # Test imports
    imports_ok, tensorflow_available = test_imports()
    if not imports_ok:
        logger.error("❌ Import tests failed. Please check your installation.")
        sys.exit(1)

    print("\n" + "-" * 40)

    # Test preprocessing
    preprocessing_ok = test_preprocessing()
    if not preprocessing_ok:
        logger.error("❌ Preprocessing tests failed.")
        sys.exit(1)

    print("\n" + "-" * 40)

    # Test CRF
    crf_ok = test_crf()
    if not crf_ok:
        logger.error("❌ CRF tests failed.")
        sys.exit(1)

    print("\n" + "-" * 40)

    # Test BiLSTM
    bilstm_ok = test_bilstm(tensorflow_available)

    print("\n" + "=" * 60)

    if preprocessing_ok and crf_ok and bilstm_ok:
        print("🎉 ALL TESTS PASSED!")
        print("\nYou can now run:")
        print("  python main.py --model crf --train    # Train CRF model")
        if tensorflow_available:
            print("  python main.py --model bilstm --train # Train BiLSTM model")
        print("  python main.py --compare               # Compare models")
        print("=" * 60)
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()