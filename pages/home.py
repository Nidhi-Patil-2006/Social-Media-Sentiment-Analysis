import streamlit as st

# --------------------------------------------------
# Page Title
# --------------------------------------------------

st.title("💬 Social Media Sentiment Analysis")
st.markdown("### Machine Learning Lab Micro-Project")

st.markdown("---")

# --------------------------------------------------
# Introduction
# --------------------------------------------------

st.header("📖 About the Project")

st.write("""
Social media has become one of the largest platforms where people express
their opinions, emotions, and experiences. Analyzing these opinions helps
organizations and researchers understand public perception towards products,
services, events, and social issues.

This project performs **Sentiment Analysis** on social media posts using
Machine Learning algorithms. The system classifies each post into one of
three sentiment categories:

- 🟢 Positive
- ⚪ Neutral
- 🔴 Negative

Two supervised Machine Learning algorithms are implemented and compared:

- Support Vector Machine (SVM)
- Random Forest
""")

st.markdown("---")

# --------------------------------------------------
# Project Objectives
# --------------------------------------------------

st.header("🎯 Project Objectives")

st.markdown("""
- Develop a sentiment classification system for social media posts.
- Perform text preprocessing and cleaning.
- Convert text into numerical features using **TF-IDF Vectorization**.
- Train Support Vector Machine and Random Forest classifiers.
- Compare both models using evaluation metrics.
- Build an interactive web application using **Streamlit**.
""")

st.markdown("---")

# --------------------------------------------------
# Dataset Information
# --------------------------------------------------

st.header("📂 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Samples", "726")

with col2:
    st.metric("Sentiment Classes", "3")

with col3:
    st.metric("Algorithms", "2")

st.write("### Sentiment Distribution")

st.markdown("""
- 🟢 Positive : **438**
- 🔴 Negative : **188**
- ⚪ Neutral : **100**
""")

st.markdown("---")

# --------------------------------------------------
# Machine Learning Pipeline
# --------------------------------------------------

st.header("⚙️ Project Workflow")

st.code("""
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Text Preprocessing
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Train-Test Split
      │
      ▼
 ┌──────────────┐
 │     SVM      │
 └──────────────┘
        │
        │
 ┌──────────────┐
 │ Random Forest│
 └──────────────┘
        │
        ▼
 Sentiment Prediction
""")

st.markdown("---")

# --------------------------------------------------
# Models Used
# --------------------------------------------------

st.header("🤖 Machine Learning Models")

left, right = st.columns(2)

with left:

    st.subheader("🧠 Support Vector Machine")

    st.write("""
- Linear SVM Classifier
- TF-IDF Features
- Balanced Class Weights
- Suitable for sparse high-dimensional text data
""")

with right:

    st.subheader("🌳 Random Forest")

    st.write("""
- Ensemble Learning Algorithm
- 500 Decision Trees
- Tuned Hyperparameters
- Balanced Class Weights
""")

st.markdown("---")

# --------------------------------------------------
# Evaluation Metrics
# --------------------------------------------------

st.header("📈 Evaluation Metrics")

st.write("""
The performance of both models is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
""")

st.markdown("---")

# --------------------------------------------------
# Technologies Used
# --------------------------------------------------

st.header("🛠 Technologies Used")

tech1, tech2, tech3 = st.columns(3)

with tech1:

    st.write("**Programming Language**")
    st.write("- Python")

with tech2:

    st.write("**Libraries**")
    st.write("""
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
""")

with tech3:

    st.write("**Framework**")
    st.write("""
- Streamlit
""")

st.markdown("---")

# --------------------------------------------------
# How to Use
# --------------------------------------------------

st.header("🚀 How to Use")

st.markdown("""
1. Open the **EDA** page to explore the dataset.
2. Navigate to **SVM Predictor** to classify a post using Support Vector Machine.
3. Navigate to **Random Forest Predictor** to classify a post using Random Forest.
4. Open the **Model Comparison** page to compare the performance of both models.
""")

st.markdown("---")

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.success("✅ Machine Learning pipeline completed successfully.")

st.caption("Developed as a Machine Learning Laboratory Micro-Project using Streamlit.")