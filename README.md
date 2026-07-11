# Social Media Sentiment Analysis

A Machine Learning-based web application for classifying social media posts into **Positive, Neutral, and Negative** sentiments.

The project implements and compares two supervised Machine Learning algorithms:

- Support Vector Machine (SVM)
- Random Forest

Textual data is preprocessed and transformed using **TF-IDF Vectorization** before model training. An interactive **Streamlit web application** is provided for Exploratory Data Analysis (EDA), sentiment prediction, and model performance comparison.

---

## Project Objective

The main objective of this project is to develop a Machine Learning system capable of analyzing the sentiment of social media posts.

The system classifies textual content into three sentiment categories:

- Positive
- Neutral
- Negative

The project also compares the performance of Support Vector Machine and Random Forest classifiers using standard Machine Learning evaluation metrics.

---

## Dataset

The project uses a Social Media Sentiment Dataset containing social media posts and their corresponding emotion or sentiment labels.

### Dataset Statistics

| Parameter | Value |
|---|---:|
| Original Records | 732 |
| Cleaned Records | 726 |
| Sentiment Classes | 3 |
| Positive Samples | 438 |
| Negative Samples | 188 |
| Neutral Samples | 100 |

The original emotion labels were mapped into three main sentiment categories: **Positive, Neutral, and Negative**.

---

## Machine Learning Workflow

```text
Raw Dataset
      |
      v
Data Preprocessing
      |
      v
Sentiment Mapping
      |
      v
Text Cleaning
      |
      v
TF-IDF Vectorization
      |
      v
Train-Test Split
      |
      +-------------------+
      |                   |
      v                   v
     SVM             Random Forest
      |                   |
      +---------+---------+
                |
                v
        Model Evaluation
                |
                v
      Sentiment Prediction
                |
                v
       Streamlit Web App
```

---

## Data Preprocessing

The following preprocessing techniques are applied to the social media text:

- Removal of unnecessary columns
- Handling of missing values
- Emotion-to-sentiment mapping
- Conversion to lowercase
- Removal of URLs
- Removal of user mentions
- Removal of hashtag symbols
- Removal of punctuation
- Removal of numbers
- Removal of stopwords
- Lemmatization

The cleaned dataset is stored as:

```text
dataset/cleaned_dataset.csv
```

---

## Feature Extraction

The project uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert textual data into numerical feature vectors.

The TF-IDF vectorizer is configured using:

```python
TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    sublinear_tf=True
)
```

### TF-IDF Configuration

- `max_features=8000` allows the vectorizer to retain up to 8000 important textual features.
- `ngram_range=(1,2)` extracts both individual words and two-word combinations.
- `sublinear_tf=True` reduces the excessive influence of highly frequent terms.

---

## Machine Learning Models

### Support Vector Machine

Support Vector Machine is used because of its effectiveness in handling high-dimensional and sparse textual data.

The SVM classifier uses a linear kernel and balanced class weights.

```python
SVC(
    kernel="linear",
    class_weight="balanced",
    random_state=42
)
```

### Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees.

The model was tuned to improve sentiment classification and minority class detection.

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

---

## Model Training

The cleaned dataset is divided into training and testing sets using an **80:20 train-test split**.

Stratified sampling is applied to preserve the sentiment class distribution.

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

---

## Model Evaluation

Both Machine Learning models are evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report
- Confusion Matrix

The initial model accuracies were:

| Model | Initial Accuracy |
|---|---:|
| Support Vector Machine | 77.40% |
| Random Forest | 75.34% |

The models were subsequently improved using TF-IDF feature configuration, balanced class weights, and Random Forest hyperparameter tuning.

Final model results are generated and stored in:

```text
outputs/metrics/comparison.csv
```

Evaluation plots are stored in:

```text
outputs/plots/
```

---

## Streamlit Application

The project includes an interactive web application developed using Streamlit.

The application consists of the following pages:

### Home

Provides an introduction to the project, objectives, dataset overview, Machine Learning workflow, and technologies used.

### Exploratory Data Analysis

Displays visual analysis of the cleaned dataset, including:

- Dataset preview
- Dataset statistics
- Sentiment distribution
- Most frequent words
- Sentence length distribution
- Positive sentiment word cloud
- Neutral sentiment word cloud
- Negative sentiment word cloud

### SVM Predictor

Allows users to enter a social media post and predict its sentiment using the trained Support Vector Machine classifier.

### Random Forest Predictor

Allows users to classify social media text using the trained Random Forest model.

### Model Comparison

Displays and compares:

- Accuracy
- Precision
- Recall
- F1-Score
- Performance comparison chart
- SVM confusion matrix
- Random Forest confusion matrix

---

## Project Structure

```text
SOCMED/
|
|-- app.py
|-- README.md
|-- requirements.txt
|-- pyproject.toml
|
|-- dataset/
|   |-- raw/
|   |   `-- sentimentdataset.csv
|   |
|   `-- cleaned_dataset.csv
|
|-- models/
|   |-- svm_model.pkl
|   |-- random_forest_model.pkl
|   |-- tfidf_vectorizer.pkl
|   |-- X_test.pkl
|   `-- y_test.pkl
|
|-- outputs/
|   |-- metrics/
|   |   |-- comparison.csv
|   |   |-- svm_metrics.txt
|   |   `-- random_forest_metrics.txt
|   |
|   `-- plots/
|       |-- model_comparison.png
|       |-- svm_confusion_matrix.png
|       `-- random_forest_confusion_matrix.png
|
|-- pages/
|   |-- home.py
|   |-- eda.py
|   |-- svm.py
|   |-- randomforest.py
|   `-- comparison.py
|
`-- src/
    |-- preprocessing.py
    |-- train_models.py
    |-- evaluate.py
    `-- predict.py
```

---

## Technologies Used

### Programming Language

- Python

### Machine Learning

- Scikit-learn

### Natural Language Processing

- NLTK
- TF-IDF Vectorization

### Data Processing

- Pandas
- NumPy

### Data Visualization

- Matplotlib
- WordCloud

### Web Application

- Streamlit

### Model Serialization

- Joblib

### Environment and Package Management

- uv

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
```

Move into the project directory:

```bash
cd socmed
```

### 2. Initialize the uv Project

If the project has not already been initialized:

```bash
uv init .
```

### 3. Install Project Dependencies

If `pyproject.toml` and `uv.lock` are available:

```bash
uv sync
```

Alternatively, dependencies can be installed from `requirements.txt`:

```bash
uv add -r requirements.txt
```

### 4. Activate the Virtual Environment

On Windows:

```bash
.venv\Scripts\activate
```

---

## Running the Machine Learning Pipeline

The complete Machine Learning pipeline can be executed in the following order.

### Step 1: Preprocess the Dataset

```bash
python src/preprocessing.py
```

This generates:

```text
dataset/cleaned_dataset.csv
```

### Step 2: Train the Models

```bash
python src/train_models.py
```

This trains the SVM and Random Forest classifiers and saves the trained models in the `models` directory.

### Step 3: Evaluate the Models

```bash
python src/evaluate.py
```

This generates evaluation metrics, comparison results, and confusion matrices.

### Step 4: Test Predictions

```bash
python src/predict.py
```

Enter a social media post when prompted.

Example:

```text
Enter a social media post: very bad movie
```

Example output:

```text
SVM Prediction:
Negative

Random Forest Prediction:
Negative
```

---

## Running the Streamlit Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in the web browser.

Use the sidebar to navigate between:

- Home
- EDA
- SVM Predictor
- Random Forest Predictor
- Model Comparison

---

## Example Predictions

### Positive Sentiment

```text
I absolutely loved the concert yesterday!
```

Expected sentiment:

```text
Positive
```

### Negative Sentiment

```text
Very bad movie
```

Expected sentiment:

```text
Negative
```

### Neutral Sentiment

```text
I wonder how this new AI model works.
```

Expected sentiment:

```text
Neutral
```

---

## Limitations

- The dataset contains a limited number of social media posts.
- The sentiment classes are moderately imbalanced.
- The Positive class contains more samples than the Negative and Neutral classes.
- TF-IDF relies on vocabulary observed during model training.
- Unseen words and expressions may be difficult for the models to classify correctly.
- Short and ambiguous social media posts may occasionally be misclassified.
- The models do not fully understand contextual meaning, sarcasm, or irony.

---

## Future Scope

The project can be further improved by:

- Increasing the size of the training dataset.
- Using a more balanced sentiment dataset.
- Applying advanced NLP techniques.
- Implementing word embeddings.
- Using deep learning models such as LSTM.
- Applying transformer-based models such as BERT.
- Performing automatic hyperparameter optimization.
- Adding real-time social media data analysis.
- Supporting multilingual sentiment analysis.
- Improving sarcasm and contextual sentiment detection.

---

## Conclusion

This project demonstrates the complete Machine Learning workflow for social media sentiment analysis. Text preprocessing and TF-IDF feature extraction were used to transform social media posts into numerical features.

Support Vector Machine and Random Forest classifiers were trained, tuned, evaluated, and compared using standard classification metrics. An interactive Streamlit application was developed to provide dataset exploration, real-time sentiment prediction, and model performance comparison.

The project highlights the importance of data preprocessing, feature extraction, class balancing, and hyperparameter tuning in the development of an effective Machine Learning-based text classification system.

---

## References

1. C. D. Manning, P. Raghavan, and H. Schütze, *Introduction to Information Retrieval*. Cambridge University Press, 2008.

2. T. Joachims, "Text Categorization with Support Vector Machines: Learning with Many Relevant Features," in *Proceedings of the European Conference on Machine Learning*, 1998.

3. L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5-32, 2001.

4. G. Salton and C. Buckley, "Term-Weighting Approaches in Automatic Text Retrieval," *Information Processing & Management*, vol. 24, no. 5, pp. 513-523, 1988.

5. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.

6. S. Bird, E. Klein, and E. Loper, *Natural Language Processing with Python*. O'Reilly Media, 2009.

---

## Project Type

Machine Learning Laboratory Micro-Project

**Topic:** Social Media Sentiment Analysis

**Models:** Support Vector Machine and Random Forest