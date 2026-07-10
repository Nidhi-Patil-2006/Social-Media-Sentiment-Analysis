import re
import string
import joblib
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --------------------------------------------------
# Download NLTK Resources
# --------------------------------------------------

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# --------------------------------------------------
# Initialize NLP Tools
# --------------------------------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# --------------------------------------------------
# Load Saved Models
# --------------------------------------------------

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

svm_model = joblib.load("models/svm_model.pkl")

rf_model = joblib.load("models/random_forest_model.pkl")

# --------------------------------------------------
# Text Cleaning Function
# --------------------------------------------------

def clean_text(text):

    if pd.isna(text):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtags
    text = text.replace("#", "")

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove emojis / non-ascii characters
    text = text.encode("ascii", "ignore").decode()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    words = []

    for word in text.split():

        if word not in stop_words:

            words.append(lemmatizer.lemmatize(word))

    return " ".join(words)

# --------------------------------------------------
# SVM Prediction
# --------------------------------------------------

def predict_svm(text):

    cleaned = clean_text(text)

    vector = vectorizer.transform([cleaned])

    prediction = svm_model.predict(vector)

    return prediction[0]

# --------------------------------------------------
# Random Forest Prediction
# --------------------------------------------------

def predict_random_forest(text):

    cleaned = clean_text(text)

    vector = vectorizer.transform([cleaned])

    prediction = rf_model.predict(vector)

    return prediction[0]

# --------------------------------------------------
# Testing
# --------------------------------------------------

if __name__ == "__main__":

    sample = input("Enter a social media post: ")

    print("\nSVM Prediction:")

    print(predict_svm(sample))

    print("\nRandom Forest Prediction:")

    print(predict_random_forest(sample))