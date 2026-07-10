import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

MODELS_DIR = "models"

OUTPUT_PLOTS = "outputs/plots"
OUTPUT_METRICS = "outputs/metrics"

os.makedirs(OUTPUT_PLOTS, exist_ok=True)
os.makedirs(OUTPUT_METRICS, exist_ok=True)

# --------------------------------------------------
# Load Saved Models
# --------------------------------------------------

print("\nLoading saved models...")

svm_model = joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl"))
rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.pkl"))

X_test = joblib.load(os.path.join(MODELS_DIR, "X_test.pkl"))
y_test = joblib.load(os.path.join(MODELS_DIR, "y_test.pkl"))

print("Models loaded successfully.")

# --------------------------------------------------
# Predictions
# --------------------------------------------------

print("\nGenerating predictions...")

svm_pred = svm_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# --------------------------------------------------
# Evaluation Function
# --------------------------------------------------

def evaluate_model(model_name, y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted"
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    report = classification_report(y_true, y_pred)

    print(f"\n========== {model_name} ==========")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:\n")
    print(report)

    with open(
        os.path.join(
            OUTPUT_METRICS,
            f"{model_name.lower().replace(' ','_')}_metrics.txt"
        ),
        "w"
    ) as file:

        file.write(f"Accuracy : {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall   : {recall:.4f}\n")
        file.write(f"F1 Score : {f1:.4f}\n\n")

        file.write(report)

    return accuracy, precision, recall, f1

# --------------------------------------------------
# Evaluate Models
# --------------------------------------------------

svm_results = evaluate_model(
    "SVM",
    y_test,
    svm_pred
)

rf_results = evaluate_model(
    "Random Forest",
    y_test,
    rf_pred
)

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

print("\nSaving confusion matrices...")

for model_name, predictions in [
    ("SVM", svm_pred),
    ("Random Forest", rf_pred)
]:

    cm = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=svm_model.classes_
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    disp.plot(ax=ax, cmap="Blues")

    plt.title(f"{model_name} Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_PLOTS,
            f"{model_name.lower().replace(' ','_')}_confusion_matrix.png"
        )
    )

    plt.close()

# --------------------------------------------------
# Comparison Table
# --------------------------------------------------

comparison = pd.DataFrame({

    "Model": ["SVM", "Random Forest"],

    "Accuracy": [
        svm_results[0],
        rf_results[0]
    ],

    "Precision": [
        svm_results[1],
        rf_results[1]
    ],

    "Recall": [
        svm_results[2],
        rf_results[2]
    ],

    "F1 Score": [
        svm_results[3],
        rf_results[3]
    ]

})

comparison.to_csv(
    os.path.join(
        OUTPUT_METRICS,
        "comparison.csv"
    ),
    index=False
)

print("\nComparison Table:\n")

print(comparison)

# --------------------------------------------------
# Accuracy Comparison Plot
# --------------------------------------------------

plt.figure(figsize=(6,5))

plt.bar(
    comparison["Model"],
    comparison["Accuracy"]
)

plt.ylim(0,1)

plt.ylabel("Accuracy")

plt.title("Model Accuracy Comparison")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PLOTS,
        "accuracy_comparison.png"
    )
)

plt.close()

print("\nEvaluation completed successfully!")