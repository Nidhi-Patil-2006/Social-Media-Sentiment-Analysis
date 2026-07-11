# 💬 Social Media Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-ML-green)
![NLTK](https://img.shields.io/badge/NLTK-NLP-yellow)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

A Machine Learning-based sentiment analysis platform designed to classify social media posts into **Positive**, **Negative**, and **Neutral** sentiments.

The project combines **Natural Language Processing (NLP)**, **Machine Learning**, and **Interactive Data Visualization** to analyze textual content and compare the performance of **Support Vector Machine (SVM)** and **Random Forest** classifiers through a Streamlit web application.

---

## 📌 Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Models Used](#models-used)
* [Technology Stack](#technology-stack)
* [Project Structure](#project-structure)
* [Workflow](#workflow)
* [Installation](#installation)
* [Using uv](#using-uv)
* [Running the ML Pipeline](#running-the-ml-pipeline)
* [Deployment](#deployment)
* [Team Members](#team-members)
* [Future Enhancements](#future-enhancements)
* [License](#license)
* [Acknowledgements](#acknowledgements)

---

## 📖 Overview

Social Media Sentiment Analysis is a Machine Learning project developed to analyze the sentiment expressed in social media posts.

The system classifies textual content into three sentiment categories:

* Positive
* Negative
* Neutral

The project implements and compares two supervised Machine Learning algorithms, **Support Vector Machine** and **Random Forest**.

The platform supports:

* Social media text preprocessing
* Emotion-to-sentiment mapping
* Exploratory Data Analysis
* TF-IDF feature extraction
* SVM-based sentiment prediction
* Random Forest-based sentiment prediction
* Model performance comparison
* Interactive Streamlit dashboard

The original dataset contains multiple emotional labels. These labels are grouped into the three primary sentiment categories to simplify the multi-class sentiment classification task.

---

## ✨ Features

### 📊 Exploratory Data Analysis

The EDA dashboard provides visual insights into the cleaned sentiment dataset.

Features include:

* Dataset preview
* Dataset summary
* Missing value analysis
* Sentiment class distribution
* Sentiment percentage analysis
* Most frequent word analysis
* Sentence length distribution
* Positive sentiment word cloud
* Neutral sentiment word cloud
* Negative sentiment word cloud

### 🤖 Sentiment Prediction

The application provides separate prediction pages for both trained Machine Learning models:

* Support Vector Machine
* Random Forest

Each model page allows users to enter a social media post and receive the predicted sentiment.

The possible prediction classes are:

* 🟢 Positive
* 🔴 Negative
* ⚪ Neutral

### 🧹 NLP Preprocessing

The preprocessing pipeline includes:

* Removal of unnecessary columns
* Missing value handling
* Emotion-to-sentiment mapping
* Text lowercasing
* URL removal
* User mention removal
* Hashtag symbol removal
* Punctuation removal
* Number removal
* Stopword removal
* Lemmatization
* TF-IDF feature extraction

### 📈 Model Comparison

The Model Comparison page provides a visual comparison of SVM and Random Forest using:

* Accuracy
* Precision
* Recall
* F1-Score
* Grouped performance comparison chart
* SVM confusion matrix
* Random Forest confusion matrix

---

## 🧠 Models Used

| Model | Type | Description |
| --- | --- | --- |
| Support Vector Machine | Machine Learning | Linear classifier suitable for sparse and high-dimensional TF-IDF text features |
| Random Forest | Machine Learning | Ensemble classifier using multiple decision trees with tuned hyperparameters |

### Support Vector Machine

SVM was selected because of its effectiveness in high-dimensional text classification problems. The model uses TF-IDF features and balanced class weights to handle the unequal sentiment class distribution.

### Random Forest

Random Forest combines multiple decision trees to perform sentiment classification. The model was tuned using increased estimators, controlled tree depth, minimum split size, minimum leaf size, and balanced class weights.

---

## 🛠️ Technology Stack

| Category | Tools |
| --- | --- |
| Programming Language | Python 3.11 |
| Web Framework | Streamlit |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, WordCloud |
| NLP | NLTK |
| Feature Extraction | TF-IDF |
| Machine Learning | Scikit-learn |
| Model Serialization | Joblib |
| Environment Management | uv |
| Deployment | Streamlit Community Cloud |

---

## 📂 Project Structure

```text
SOCMED/
│
├── app.py
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── README.md
│
├── dataset/
│   ├── raw/
│   │   └── sentimentdataset.csv
│   │
│   └── cleaned_dataset.csv
│
├── models/
│   ├── svm_model.pkl
│   ├── random_forest_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── X_test.pkl
│   └── y_test.pkl
│
├── outputs/
│   ├── metrics/
│   │   └── comparison.csv
│   │
│   └── plots/
│       ├── model_comparison.png
│       ├── svm_confusion_matrix.png
│       └── random_forest_confusion_matrix.png
│
├── pages/
│   ├── home.py
│   ├── eda.py
│   ├── svm.py
│   ├── randomforest.py
│   └── comparison.py
│
└── src/
    ├── preprocessing.py
    ├── train_models.py
    ├── evaluate.py
    └── predict.py
```

---

## 🔄 Workflow

1. Load the social media sentiment dataset
2. Remove unnecessary dataset attributes
3. Handle missing values
4. Map multiple emotion labels into Positive, Negative, and Neutral sentiments
5. Clean and preprocess social media text
6. Apply stopword removal and lemmatization
7. Convert textual data into numerical TF-IDF features
8. Split the dataset into training and testing sets
9. Train Support Vector Machine and Random Forest models
10. Tune model parameters and handle class imbalance
11. Evaluate models using standard classification metrics
12. Save trained models and the TF-IDF vectorizer
13. Build the Streamlit dashboard
14. Perform interactive sentiment prediction and model comparison

---

## 📊 Dataset

The original social media sentiment dataset contains **732 records** and **15 attributes**.

After preprocessing, the final dataset contains **726 valid records** with two required attributes:

* Text
* Sentiment

### Sentiment Distribution

| Sentiment | Number of Samples |
| --- | ---: |
| Positive | 438 |
| Negative | 188 |
| Neutral | 100 |
| **Total** | **726** |

The original emotion labels were mapped into the three main sentiment categories: **Positive**, **Negative**, and **Neutral**.

---

## 🔤 TF-IDF Feature Extraction

TF-IDF Vectorization is used to convert the cleaned social media text into numerical features.

The vectorizer is configured using:

```python
TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    sublinear_tf=True
)
```

The configuration supports:

* Up to 8000 textual features
* Unigram feature extraction
* Bigram feature extraction
* Sublinear term frequency scaling

This allows the models to identify important individual words and two-word combinations within social media posts.

---

## 📈 Model Development and Improvement

During initial training, the models achieved the following accuracy:

| Model | Initial Accuracy |
| --- | ---: |
| Support Vector Machine | 77.40% |
| Random Forest | 75.34% |

Initially, SVM performed better than Random Forest for TF-IDF-based text classification.

The models were further improved using:

* Increased TF-IDF feature size
* Sublinear TF scaling
* Balanced class weights
* Increased number of Random Forest trees
* Maximum tree depth control
* Minimum split sample tuning
* Minimum leaf sample tuning

The tuned Random Forest model uses:

```python
RandomForestClassifier(
    n_estimators=500,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)
```

The final performance of both models is compared using Accuracy, Precision, Recall, and F1-Score.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <repository-url>
cd socmed
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment on Windows:

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## ⚡ Using uv

The project uses `uv` for Python environment and dependency management.

Initialize the project:

```bash
uv init .
```

Install dependencies from `requirements.txt`:

```bash
uv add -r requirements.txt
```

Alternatively, if `pyproject.toml` and `uv.lock` are available:

```bash
uv sync
```

Activate the virtual environment on Windows:

```bash
.venv\Scripts\activate
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 🧪 Running the ML Pipeline

### Step 1: Data Preprocessing

Run:

```bash
python src/preprocessing.py
```

This generates the cleaned dataset:

```text
dataset/cleaned_dataset.csv
```

### Step 2: Model Training

Run:

```bash
python src/train_models.py
```

This trains the SVM and Random Forest classifiers and saves the trained models and TF-IDF vectorizer.

### Step 3: Model Evaluation

Run:

```bash
python src/evaluate.py
```

This generates:

* Performance metrics
* Model comparison results
* Grouped performance chart
* SVM confusion matrix
* Random Forest confusion matrix

### Step 4: Test Sentiment Prediction

Run:

```bash
python src/predict.py
```

Example input:

```text
very bad movie
```

Example output:

```text
SVM Prediction:
Negative

Random Forest Prediction:
Negative
```

### Step 5: Run the Streamlit Application

Run:

```bash
streamlit run app.py
```

Use the sidebar to navigate between:

* Home
* EDA
* SVM Predictor
* Random Forest Predictor
* Model Comparison

---

## 🚀 Deployment

The application can be deployed using **Streamlit Community Cloud**.

🔗 **Deployment URL:** `<Add Streamlit Deployment URL>`

The main files required for deployment include:

```text
app.py
requirements.txt
pages/
models/
dataset/
outputs/
src/
```

---

## 👥 Team Members

| Team Member | GitHub | Responsibilities |
| --- | --- | --- |
| Nidhi Patil | https://github.com/Nidhi-Patil-2006 | Data preprocessing, sentiment mapping and documentation, Random Forest tuning |
| Prisha Sawant | https://github.com/prisha-s24 | SVM model development and evaluation, Streamlit application |

---

## 🌱 Future Enhancements

* Increase the size of the social media training dataset
* Improve class balance between sentiment categories
* Implement multilingual sentiment analysis
* Add confidence scores to model predictions
* Improve sarcasm and irony detection
* Use word embedding techniques
* Add real-time social media API integration

---
