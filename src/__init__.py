"""
Quranic Named Entity Recognition Package

This package provides implementations of BiLSTM and CRF models
for Named Entity Recognition on Quranic text.

Modules:
- preprocessing: Data loading, cleaning, and preprocessing utilities
- bilstm: Bidirectional LSTM model implementation
- crf: Conditional Random Field model implementation
"""

__version__ = "1.0.0"
__author__ = "Quranic NER Team"
__description__ = "Named Entity Recognition for Quranic Text using BiLSTM and CRF"

try:
    from .preprocessing import QuranicNERPreprocessor, create_data_splits
    from .bilstm import QuranicBiLSTM, create_bilstm_model
    from .crf import QuranicCRF, create_crf_model
    __all__ = [
        'QuranicNERPreprocessor',
        'QuranicBiLSTM',
        'QuranicCRF',
        'create_data_splits',
        'create_bilstm_model',
        'create_crf_model'
    ]
except ImportError:
    # Fallback for when TensorFlow is not available
    try:
        from .preprocessing import QuranicNERPreprocessor, create_data_splits
        from .crf import QuranicCRF, create_crf_model
        __all__ = [
            'QuranicNERPreprocessor',
            'QuranicCRF',
            'create_data_splits',
            'create_crf_model'
        ]
        print("Warning: BiLSTM modules not available. CRF functionality only.")
    except ImportError as e:
        raise ImportError(f"Critical modules not available: {e}")