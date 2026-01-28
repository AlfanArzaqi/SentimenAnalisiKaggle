# Sentiment Analysis - Twitter Dataset

## 📊 Project Overview

Implementasi analisis sentimen menggunakan 3 model berbeda dengan target akurasi >92% pada training dan testing set.

## 🎯 Objectives

- **Dataset**: Twitter Entity Sentiment Analysis (74,682 samples)
- **Classes**: 3 classes (Positive, Negative, Neutral)
- **Target**: Training & Testing Accuracy > 92%
- **Models**: 3 different approaches

## 🤖 Models

### 1. Optimized Logistic Regression + TF-IDF
- **Feature Extraction**: TF-IDF with unigram + bigram
- **Data Split**: 80/20
- **Approach**: Traditional Machine Learning

### 2. BiLSTM + Attention Mechanism
- **Feature Extraction**: Trainable Embedding (128-dim)
- **Data Split**: 80/20
- **Architecture**: Bidirectional LSTM with Bahdanau Attention

### 3. Multi-Filter CNN
- **Feature Extraction**: Trainable Embedding (128-dim)
- **Data Split**: 70/30
- **Architecture**: CNN with multiple filter sizes [2, 3, 4, 5]

## 📁 Project Structure

```
SentimenAnalisiKaggle/
├── Sentiment_Analysis_Complete.ipynb   # Main notebook (11 cells)
├── requirements.txt                     # Dependencies
└── README.md                            # Documentation
```

## 🚀 How to Run

### For Graders/Reviewers (Google Colab - Recommended)
1. Open `Sentiment_Analysis_Complete.ipynb` in Google Colab
2. Click "Runtime" → "Run all"
3. Dataset loads automatically from repository
4. No additional setup required!

### For Local Development
```bash
git clone https://github.com/AlfanArzaqi/SentimenAnalisiKaggle.git
cd SentimenAnalisiKaggle
pip install -r requirements.txt
jupyter notebook Sentiment_Analysis_Complete.ipynb
```

### Dataset
- Pre-downloaded and included in `dataset/` folder
- Source: Kaggle Twitter Entity Sentiment Analysis
- No API keys or credentials needed

### Features
- ✅ Auto-detects Google Colab vs Local environment
- ✅ Automatic repository cloning in Colab
- ✅ Path handling for cross-platform compatibility
- ✅ Clear error messages and file verification

## 📝 Notebook Structure

| Cell | Section | Description |
|------|---------|-------------|
| 1 | Import Libraries | Import all dependencies and setup |
| 2 | Load Data | Load and display dataset |
| 3 | Data Exploration | EDA with visualizations |
| 4 | Preprocessing | Complete text preprocessing pipeline |
| 5 | Prepare Data | Label encoding and feature extraction |
| 6 | Model 1 | Logistic Regression + TF-IDF (complete) |
| 7 | Model 2 | BiLSTM + Attention (complete) |
| 8 | Model 3 | Multi-Filter CNN (complete) |
| 9 | Comparison | Model comparison and target check |
| 10 | Inference | Testing with new samples |
| 11 | Save Models | Export all trained models |

## 🔧 Preprocessing Pipeline

1. **Lowercase** - Convert to lowercase
2. **Remove URLs** - Remove all URLs and links
3. **Remove Mentions** - Remove @mentions
4. **Remove Retweets** - Remove RT markers
5. **Remove Special Chars** - Clean special characters
6. **Tokenization** - Split into tokens
7. **Remove Stopwords** - Remove English stopwords
8. **Stemming** - SnowballStemmer for English

## 📊 Expected Results

| Model | Train Acc | Test Acc | Data Split |
|-------|-----------|----------|------------|
| Logistic Regression | >92% | >92% | 80/20 |
| BiLSTM + Attention | >92% | >92% | 80/20 |
| Multi-Filter CNN | >92% | >92% | 70/30 |

## 🔍 Model Differences

### Feature Extraction
- **Logistic Regression**: TF-IDF (max_features=10000, ngram_range=(1,2))
- **BiLSTM & CNN**: Trainable Embedding (vocab_size=10000, embed_dim=128)

### Data Split
- **LR & BiLSTM**: 80% training, 20% testing
- **CNN**: 70% training, 30% testing

### Architecture
- **LR**: Linear classifier
- **BiLSTM**: Bidirectional LSTM with Attention mechanism
- **CNN**: Parallel CNN filters with different kernel sizes

## 💾 Saved Models

After running the notebook, following files will be generated:

- `logistic_regression_model.pkl` - LR model
- `tfidf_vectorizer.pkl` - TF-IDF vectorizer
- `bilstm_attention_model.h5` - BiLSTM model
- `multi_filter_cnn_model.h5` - CNN model
- `tokenizer.json` - Keras tokenizer
- `label_encoder.pkl` - Label encoder

## 📈 Visualizations

The notebook includes:
- Sentiment distribution plots
- Training history (accuracy & loss curves)
- Confusion matrices for all models
- Model comparison bar charts

## 🧪 Inference Examples

The notebook tests 5 sample sentences:
1. Positive review
2. Negative review
3. Neutral review
4. Strong positive
5. Strong negative

Each sample is predicted by all 3 models with confidence scores.

## 🛠️ Dependencies

**Minimum versions** (see requirements.txt for details):
- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- nltk >= 3.6.0
- tensorflow >= 2.8.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- tqdm >= 4.62.0

## 📚 Dataset Information

**Source**: [Kaggle - Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)

**Location**: `dataset/` folder (included in repository)

**Files**:
- `twitter_training.csv` - Main training dataset (74,682 samples)
- `twitter_validation.csv` - Validation dataset

**Original Classes**: Positive, Negative, Neutral, Irrelevant

**Processed Classes**: Positive, Negative, Neutral (Irrelevant merged to Neutral)

**Total Samples**: 74,682 tweets (after preprocessing: ~74,000)

## ✅ Success Criteria

- ✅ All 3 models trained successfully
- ✅ At least 1 model achieves >92% accuracy
- ✅ Complete preprocessing pipeline
- ✅ Visualizations included
- ✅ Inference examples working
- ✅ Models saved for reuse

## 📖 References

- [Twitter Entity Sentiment Analysis Dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
- [NLTK Documentation](https://www.nltk.org/)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

## 👤 Author

**Alfan Arzaqi**

## 📄 License

This project is for educational purposes.

---

**Note**: The dataset is included in the repository's `dataset/` folder - no additional download required!