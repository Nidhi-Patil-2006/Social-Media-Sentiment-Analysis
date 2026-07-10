import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# --------------------------------------------------
# File Paths
# --------------------------------------------------

DATASET_PATH = "dataset/cleaned_dataset.csv"

MODELS_DIR = "models"

SVM_MODEL_PATH = os.path.join(MODELS_DIR, "svm_model.pkl")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

X_TEST_PATH = os.path.join(MODELS_DIR, "X_test.pkl")
Y_TEST_PATH = os.path.join(MODELS_DIR, "y_test.pkl")

# --------------------------------------------------
# Create models folder if it doesn't exist
# --------------------------------------------------

os.makedirs(MODELS_DIR, exist_ok=True)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

print("\nLoading cleaned dataset...")

df = pd.read_csv(DATASET_PATH)

print(f"Dataset Shape : {df.shape}")

# --------------------------------------------------
# Features and Labels
# --------------------------------------------------

X = df["Text"]
y = df["Sentiment"]

# --------------------------------------------------
# TF-IDF Vectorization
# --------------------------------------------------

print("\nApplying TF-IDF Vectorization...")

vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X = vectorizer.fit_transform(X)

print(f"Feature Matrix Shape : {X.shape}")

# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------

print("\nSplitting Dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training Samples : {X_train.shape[0]}")
print(f"Testing Samples  : {X_test.shape[0]}")

# --------------------------------------------------
# Train SVM
# --------------------------------------------------

print("\nTraining Support Vector Machine...")

svm_model = SVC(
    kernel="linear",
    C=5,
    class_weight="balanced",
    random_state=42
)

svm_model.fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_predictions)

print(f"SVM Accuracy : {svm_accuracy:.4f}")

# --------------------------------------------------
# Train Random Forest
# --------------------------------------------------

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)

print(f"Random Forest Accuracy : {rf_accuracy:.4f}")

# --------------------------------------------------
# Save Models
# --------------------------------------------------

print("\nSaving models...")

joblib.dump(vectorizer, VECTORIZER_PATH)

joblib.dump(svm_model, SVM_MODEL_PATH)

joblib.dump(rf_model, RF_MODEL_PATH)

joblib.dump(X_test, X_TEST_PATH)

joblib.dump(y_test, Y_TEST_PATH)

print("Models saved successfully.")

# --------------------------------------------------
# Training Summary
# --------------------------------------------------

print("\n========== TRAINING SUMMARY ==========")

print(f"Dataset Size          : {len(df)}")

print(f"Training Samples      : {X_train.shape[0]}")

print(f"Testing Samples       : {X_test.shape[0]}")

print(f"SVM Accuracy          : {svm_accuracy:.4f}")

print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

print("\nSaved Files:")

print(f"✓ {SVM_MODEL_PATH}")

print(f"✓ {RF_MODEL_PATH}")

print(f"✓ {VECTORIZER_PATH}")

print(f"✓ {X_TEST_PATH}")

print(f"✓ {Y_TEST_PATH}")

print("\nTraining completed successfully!")