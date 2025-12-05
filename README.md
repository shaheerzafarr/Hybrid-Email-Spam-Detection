# 📧 Hybrid Email Spam Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io)
[![ML](https://img.shields.io/badge/ML-Extra%20Trees-orange.svg)]()

A robust, hybrid spam detection engine that combines **Machine Learning (Extra Trees Classifier)** with **Graph-Based Similarity Networks** to detect sophisticated spam campaigns. Deployed as an interactive web application using Streamlit.

---

## 🎯 Overview

Traditional spam filters often rely solely on keyword probability or static rules. This project introduces a **Hybrid Framework** that analyzes emails in two dimensions:

### 1. **Content Analysis** 
Using Natural Language Processing (NLP) and Machine Learning to detect spam patterns in text.

### 2. **Relational Analysis**
Using Cosine Similarity and Graph Theory to detect if an incoming email is structurally similar to known spam clusters, even if specific keywords are obfuscated.

---

## ✨ Features

- **Hybrid Scoring Engine**: Combines ML probability ($P_{ML}$) and Graph Similarity ($S_{Sim}$) using a weighted formula:

  $$\text{Score} = (0.6 \times P_{ML}) + (0.4 \times S_{Sim})$$

- **Advanced Preprocessing**: Includes tokenization, stemming (PorterStemmer), and stopword removal

- **Feature Engineering**: Merges TF-IDF textual vectors with structural metadata (character counts, word counts)

- **Interactive Visualization**: Dynamically generates a Similarity Graph showing how the incoming email connects to the nearest neighbors in the dataset

- **Real-Time Feedback**: Instant classification with a detailed breakdown of ML confidence vs. Similarity score

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **Web Framework** | Streamlit |
| **Machine Learning** | Scikit-Learn (Extra Trees Classifier), XGBoost |
| **NLP** | NLTK |
| **Graph Theory** | NetworkX |
| **Data Processing** | Pandas, NumPy, SciPy |
| **Visualization** | Matplotlib, Seaborn |

---

## 📁 Project Structure

```
hybrid-spam-detection/
│
├── app.py                      # Main Streamlit application file
├── SpamDetection.ipynb         # Jupyter Notebook for EDA, training, and evaluation
├── model.pkl                   # Trained Extra Trees Classifier (generated via notebook)
├── vectorizer.pkl              # Fitted TF-IDF Vectorizer
├── similarity_features.pkl     # Processed dataset for graph comparison
├── requirements.txt            # List of dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Installation & Setup

### Prerequisites

Ensure you have **Python 3.8+** installed on your system.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/hybrid-spam-detection.git
cd hybrid-spam-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, install the core libraries manually:

```bash
pip install streamlit pandas numpy nltk scikit-learn networkx matplotlib scipy
```

### 3. Download NLTK Data

```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will launch in your default browser at `http://localhost:8501`

---

## 📊 How It Works

### Workflow Pipeline

```
📥 User Input (Email Text)
         ↓
🔧 Preprocessing (Clean, Tokenize, Stem)
         ↓
📊 Feature Extraction (TF-IDF + Structural Features)
         ↓
    ┌────┴────┐
    ↓         ↓
🤖 ML Model  📈 Graph Similarity
    ↓         ↓
    └────┬────┘
         ↓
⚖️ Weighted Ensemble Score
         ↓
✅ Classification Result (SPAM/HAM)
```

### Step-by-Step Process

1. **Input**: User pastes an email body into the text area

2. **Preprocessing**: The text is cleaned, stemmed, and vectorized (TF-IDF)

3. **ML Prediction**: The Extra Trees Classifier predicts the probability of the email being spam based on trained patterns from the Enron dataset

4. **Graph Analysis**: The system calculates Cosine Similarity between the input vector and thousands of stored emails. It identifies the "neighborhood" of the new email

5. **Decision**:
   - If the email looks like spam (ML) **OR** is hidden in a cluster of known spam (Graph), the Weighted Score increases
   - **Threshold for Spam**: > 0.5

---

## 🖼️ Screenshots

### 1. Dashboard Interface
<img width="2475" height="1135" alt="image" src="https://github.com/user-attachments/assets/384864a0-49f2-4536-a6a6-6b8a948c2ea0" />

### 2. Similarity Graph Visualization
<img width="667" height="442" alt="image" src="https://github.com/user-attachments/assets/f1a5c0fe-c610-40d8-92c0-af04bd343552" />

*Interactive network graph showing email relationships and spam clusters*


---

## 🔬 Model Performance

During training on the **Enron Spam Dataset**, the models achieved the following performance:

| Model | Accuracy | Precision | Status |
|-------|----------|-----------|--------|
| **Extra Trees** ⭐ | **98.8%** | **High** | ✅ Selected |
| Random Forest | 98.5% | High | - |
| Naive Bayes | 96.5% | Moderate | - |

### Why Extra Trees?

The **Extra Trees** model was selected for its:
- **Superior precision**: Minimizing the risk of flagging legitimate emails as spam
- **Robustness**: Random split points reduce overfitting
- **Speed**: Fast training and prediction times
- **Noise resistance**: Handles variations in spam patterns effectively

---

## 📈 Performance Metrics

### Confusion Matrix Results
- **True Positives**: High detection of actual spam
- **True Negatives**: Accurate identification of legitimate emails
- **False Positives**: Minimized (critical for user experience)
- **False Negatives**: Low miss rate on spam detection

### Classification Report
```
              precision    recall  f1-score   support

         Ham       0.99      0.99      0.99      3500
        Spam       0.99      0.98      0.98      3200

    accuracy                           0.99      6700
   macro avg       0.99      0.99      0.99      6700
weighted avg       0.99      0.99      0.99      6700
```

---

## 🎓 Dataset Information

**Enron Spam Dataset**
- **Source**: Carnegie Mellon University
- **Size**: ~33,000 emails
- **Classes**: Binary (Spam/Ham)
- **Features**: Email text, subject lines, metadata
- **Distribution**: Balanced dataset for unbiased training

---

## 🔧 Configuration

### Adjusting the Ensemble Weights

You can modify the hybrid scoring weights in `app.py`:

```python
# Current configuration
alpha = 0.6  # Weight for ML probability
beta = 0.4   # Weight for Similarity score

# Adjust based on your preference:
# - Higher alpha: Trust ML model more
# - Higher beta: Trust graph similarity more
```

### Similarity Threshold

Adjust the spam cluster threshold:

```python
SIMILARITY_THRESHOLD = 0.8  # Default: 80% similarity
```

---

## 🚦 Usage Example

```python
# Example email input
email_text = """
Congratulations! You've won $1,000,000!
Click here immediately to claim your prize.
This offer expires in 24 hours!
"""

# System processes the email and returns:
{
    "classification": "SPAM",
    "ml_score": 0.95,
    "similarity_score": 0.87,
    "final_score": 0.92,
    "confidence": "High"
}
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Steps to Contribute

1. **Fork the Project**
   ```bash
   git fork https://github.com/your-username/hybrid-spam-detection.git
   ```

2. **Create your Feature Branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```

3. **Commit your Changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```

4. **Push to the Branch**
   ```bash
   git push origin feature/AmazingFeature
   ```

5. **Open a Pull Request**

### Contribution Ideas
- Add support for additional languages
- Implement email attachment scanning
- Create a REST API version
- Add more visualization options
- Improve preprocessing pipeline
- Optimize graph similarity algorithm

---

## 🛣️ Roadmap

- [ ] **Phase 1**: Add batch email processing
- [ ] **Phase 2**: Implement deep learning models (BERT/RoBERTa)
- [ ] **Phase 3**: Create REST API with FastAPI
- [ ] **Phase 4**: Develop browser extension for Gmail/Outlook
- [ ] **Phase 5**: Mobile application (React Native)
- [ ] **Phase 6**: Real-time database updates with user feedback

---

## 👨‍💻 Author

- GitHub: [shaheerzafarr]([https://github.com/your-username](https://github.com/shaheerzafarr))
- LinkedIn: [Muhammad Shaheer Zafar]([https://linkedin.com/in/yourprofile](https://www.linkedin.com/in/muhammad-shaheer-zafar-146a08356/))
- Email: shaheerzafar2004@gmail.com

## 🙏 Acknowledgments

- **Enron Dataset**: Carnegie Mellon University for providing the dataset
- **Scikit-learn Team**: For the excellent ML library
- **NLTK Contributors**: For comprehensive NLP tools
- **Streamlit**: For the intuitive web framework
- **NetworkX**: For powerful graph analysis capabilities
- **Open Source Community**: For continuous inspiration and support

---

## 📚 References

1. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

2. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media.

3. Enron Email Dataset. Carnegie Mellon University. Available at: [https://www.cs.cmu.edu/~enron/](https://www.cs.cmu.edu/~enron/)

4. Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. *Machine Learning*, 63(1), 3-42.

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is not intended to replace professional email security solutions. Always use multiple layers of security when dealing with sensitive communications.

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐!

[![Star History Chart](https://api.star-history.com/svg?repos=shaheerzafarr/hybrid-spam-detection&type=Date)](https://star-history.com/#shaheerzafarr/hybrid-spam-detection&Date)

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/your-username/hybrid-spam-detection?style=social)
![GitHub forks](https://img.shields.io/github/forks/your-username/hybrid-spam-detection?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/your-username/hybrid-spam-detection?style=social)
![GitHub issues](https://img.shields.io/github/issues/your-username/hybrid-spam-detection)
![GitHub pull requests](https://img.shields.io/github/issues-pr/your-username/hybrid-spam-detection)

---

<div align="center">

**Made with ❤️ and Python**

[⬆ Back to Top](#-hybrid-email-spam-detection-system)

</div>
