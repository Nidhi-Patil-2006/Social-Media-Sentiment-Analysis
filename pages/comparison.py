import streamlit as st
import pandas as pd
from PIL import Image

# --------------------------------------------------
# Page Title
# --------------------------------------------------

st.title("📈 Model Comparison")

st.markdown("""
Comparison of the performance of **Support Vector Machine (SVM)** and
**Random Forest** classifiers.
""")

st.divider()

# --------------------------------------------------
# Load Comparison Data
# --------------------------------------------------

comparison = pd.read_csv("outputs/metrics/comparison.csv")

# --------------------------------------------------
# Performance Metrics
# --------------------------------------------------

st.header("📊 Performance Metrics")

col1, col2 = st.columns(2)

# -------------------------
# SVM Metrics
# -------------------------

with col1:

    st.subheader("🧠 SVM")

    st.metric(
        "Accuracy",
        f"{comparison.loc[0,'Accuracy']:.2%}"
    )

    st.metric(
        "Precision",
        f"{comparison.loc[0,'Precision']:.2%}"
    )

    st.metric(
        "Recall",
        f"{comparison.loc[0,'Recall']:.2%}"
    )

    st.metric(
        "F1 Score",
        f"{comparison.loc[0,'F1 Score']:.2%}"
    )

# -------------------------
# Random Forest Metrics
# -------------------------

with col2:

    st.subheader("🌳 Random Forest")

    st.metric(
        "Accuracy",
        f"{comparison.loc[1,'Accuracy']:.2%}"
    )

    st.metric(
        "Precision",
        f"{comparison.loc[1,'Precision']:.2%}"
    )

    st.metric(
        "Recall",
        f"{comparison.loc[1,'Recall']:.2%}"
    )

    st.metric(
        "F1 Score",
        f"{comparison.loc[1,'F1 Score']:.2%}"
    )

st.divider()

# --------------------------------------------------
# Comparison Table
# --------------------------------------------------

st.header("📋 Comparison Table")

st.dataframe(
    comparison,
    use_container_width=True
)

st.divider()

# --------------------------------------------------
# Accuracy Comparison
# --------------------------------------------------

st.header("📈 Accuracy Comparison")

comparison_plot = Image.open(
    "outputs/plots/model_comparison.png"
)

st.image(
    comparison_plot,
    use_container_width=True
)

st.divider()

# --------------------------------------------------
# Confusion Matrices
# --------------------------------------------------

st.header("🧩 Confusion Matrices")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Support Vector Machine")

    svm_cm = Image.open(
        "outputs/plots/svm_confusion_matrix.png"
    )

    st.image(
        svm_cm,
        use_container_width=True
    )

with col2:

    st.subheader("Random Forest")

    rf_cm = Image.open(
        "outputs/plots/random_forest_confusion_matrix.png"
    )

    st.image(
        rf_cm,
        use_container_width=True
    )

st.divider()

# --------------------------------------------------
# Best Model
# --------------------------------------------------

st.header("🏆 Best Performing Model")

best_model = comparison.loc[
    comparison["Accuracy"].idxmax(),
    "Model"
]

best_accuracy = comparison["Accuracy"].max()

st.success(
    f"Best Model: **{best_model}**\n\n"
    f"Accuracy: **{best_accuracy:.2%}**"
)

st.divider()

# --------------------------------------------------
# Conclusion
# --------------------------------------------------

st.header("📝 Conclusion")

st.write("""
Both machine learning models successfully classified social media posts
into Positive, Neutral and Negative sentiments.

The comparison above highlights their performance based on Accuracy,
Precision, Recall and F1 Score.

The model with the highest overall accuracy is identified as the best
performing classifier for this dataset.
""")