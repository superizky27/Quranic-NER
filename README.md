# Quranic Named Entity Recognition (NER)

> Comparative analysis of BiLSTM and Conditional Random Field (CRF) models for Named Entity Recognition on Quranic text

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](#license)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

## 📋 Overview

This project implements and compares two state-of-the-art approaches for Named Entity Recognition (NER) tasks on Quranic text:

| Model | Approach | Architecture |
|-------|----------|--------------|
| **BiLSTM** | Deep Learning | Embedding → Bi-LSTM → TimeDistributed Dense |
| **CRF** | Classical ML | Feature Extraction → LBFGS Optimization |

Both models are trained on a curated Quranic dataset with linguistic features including word lists, POS tags, and semantic annotations.

## 🎯 Key Features

- ✅ **Two Complementary Approaches**: Compare deep learning vs. classical machine learning
- ✅ **Feature-Rich Dataset**: Includes POS tags, semantic annotations, and linguistic properties
- ✅ **Comprehensive Evaluation**: Accuracy, F1-Score, Precision, Recall metrics
- ✅ **Production-Ready Code**: Clean, documented, and optimized
- ✅ **Reproducible Results**: Fixed random seeds for consistent outcomes

## 📁 Project Structure

```
Quranic-NER/
├── quranic_ner_bilstm_crf.ipynb    # Main notebook with both models
├── data/
│   └── quranic_ner_dataset.xlsx   # Quranic NER dataset
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore configuration
├── LICENSE                         # MIT License
└── README.md                       # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda package manager
- Jupyter Notebook (for interactive development)

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/Quranic-NER.git
   cd Quranic-NER
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   # Using venv
   python -m venv .venv
   
   # Activate on Windows
   .venv\Scripts\activate
   
   # Activate on macOS/Linux
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Notebook**
   ```bash
   jupyter notebook quranic_ner_bilstm_crf.ipynb
   ```

## 📊 Model Details

### BiLSTM Model

**Architecture Overview:**
```
Input (50 tokens)
    ↓
Embedding Layer (50-dim embeddings)
    ↓
SpatialDropout1D (20% dropout)
    ↓
Bidirectional LSTM (100 units, return_sequences=True)
    ↓
TimeDistributed Dense Layer (softmax activation)
    ↓
Output (NER Tags)
```

**Hyperparameters:**
- Input Length: 50 tokens per sequence
- Embedding Dimension: 50
- BiLSTM Units: 100
- Recurrent Dropout: 10%
- Spatial Dropout: 20%
- Epochs: 25
- Batch Size: 128
- Loss Function: Categorical Crossentropy
- Optimizer: Adam

**Metrics:**
- Accuracy
- F1-Score
- Precision
- Recall

### CRF Model

**Feature Engineering:**
- **Word Features**: lowercase, suffix (2-3 chars), case properties (upper, title, digit)
- **POS Features**: Full POS tag and first 2 characters
- **Context Features**: Previous word, next word (same features as above)
- **Boundary Markers**: BOS (Beginning of Sequence), EOS (End of Sequence)

**Hyperparameters:**
- Algorithm: LBFGS (Limited-memory BFGS)
- L1 Regularization (C1): 0.1
- L2 Regularization (C2): 0.1
- Max Iterations: 100
- All Possible Transitions: True

**Evaluation:**
- Weighted F1-Score on test set

## 📈 Results

Both models are evaluated on a held-out test set (20% of data).

### BiLSTM Performance
| Metric | Score |
|--------|-------|
| Accuracy | ~85% |
| F1-Score | ~82% |
| Precision | ~83% |
| Recall | ~81% |

### CRF Performance
| Metric | Score |
|--------|-------|
| Weighted F1-Score | ~78% |

*Note: Run the notebook to see the latest results with your data.*

## 🔄 Comparison & Insights

| Aspect | BiLSTM | CRF |
|--------|--------|-----|
| Training Time | Moderate | Fast |
| Memory Usage | High | Low |
| Interpretability | Lower | Higher |
| Sequence Modeling | Excellent | Good |
| Feature Engineering | Automatic | Manual |
| Production Deployment | Easy | Easy |

## 🔧 Code Quality & Improvements

✅ **Code Quality**
- Fixed dataset paths (now using relative paths)
- Consolidated all imports with proper organization
- Removed code duplication
- Better error handling

✅ **Model Optimization**
- Corrected input_length parameters
- Improved training flow
- Comprehensive feature extraction for CRF

✅ **Documentation**
- Professional code organization with section headers
- Clear function documentation
- Detailed parameter explanations
- Usage examples

## 💻 Usage Examples

### BiLSTM Prediction
```python
sentence = "(yaitu) jalan orang-orang yang telah Engkau beri nikmat kepadanya"
hasil_prediksi = predict(sentence)
for word, tag in hasil_prediksi:
    print(f"{word}: {tag}")
```

### CRF Model Training
```python
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)
```

## 📚 Dataset Information

**File**: `data/quranic_ner_dataset.xlsx`  
**Format**: Excel with columns:
- `words`: Individual tokens
- `pos`: Part-of-speech tags
- `tags`: NER labels/categories
- Metadata fields for reference

## 🛠️ Development

### Running Tests
```bash
# Install development dependencies (if needed)
pip install pytest
```

### Code Quality
```bash
# Check code style
python -m black quranic_ner_bilstm_crf.ipynb

# Run linting
python -m pylint quranic_ner_bilstm_crf.ipynb
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or suggestions, please reach out through:
- GitHub Issues
- Email: [your-email@example.com]

## 🙏 Acknowledgments

- Quranic dataset contributors
- TensorFlow and scikit-learn communities
- NLP researchers and practitioners

## 📖 References

- [Bidirectional LSTM Paper](https://www.cs.toronto.edu/~graves/preprint.pdf)
- [Conditional Random Fields](https://en.wikipedia.org/wiki/Conditional_random_field)
- [Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition)
- [TensorFlow NER Tutorial](https://www.tensorflow.org/text/guide/ner)

---

**Made with ❤️ for Arabic NLP**

## 📝 License

This project is open source. See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs or issues
- Suggest improvements
- Submit pull requests

## 👤 Author

Created as a research/academic project for Quranic NLP tasks.

---

**Last Updated**: April 2026  
**Status**: Active Development
