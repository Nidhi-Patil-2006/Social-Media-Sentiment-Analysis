import streamlit as st

from src.predict import predict_svm

# --------------------------------------------------
# Page Title
# --------------------------------------------------

st.title("🧠 Support Vector Machine Predictor")


st.markdown("""
### About the Model

Support Vector Machine (SVM) is a supervised machine learning algorithm used for text classification. 
In this project, SVM classifies social media posts as **Positive**, **Negative**, or **Neutral** based on their textual content.

#### Features
- ✅ Linear SVM Classifier
- ✅ TF-IDF Feature Extraction
- ✅ Balanced Class Weights
- ✅ Suitable for sparse, high-dimensional text data
""")

st.divider()
st.markdown("""
Enter a social media post below to predict its sentiment using the
**Support Vector Machine (SVM)** model.
""")

st.divider()

# --------------------------------------------------
# Text Input
# --------------------------------------------------

user_input = st.text_area(
    "Enter Social Media Post",
    height=180,
    placeholder="Example: I absolutely loved the concert yesterday!"
)

# --------------------------------------------------
# Prediction Button
# --------------------------------------------------

if st.button("Predict Sentiment", use_container_width=True):

    if user_input.strip() == "":

        st.warning("Please enter some text.")

    else:

        prediction = predict_svm(user_input)

        st.subheader("Prediction")

        if prediction == "Positive":

            st.success("🟢 Positive\n\n"
                       "The post expresses a favourable or optimistic sentiment.")

        elif prediction == "Negative":

            st.error("🔴 Negative\n\n"
                     "The post expresses an unfavourable or negative sentiment.")

        else:

            st.info("⚪ Neutral\n\n"
                    "The post is informational or does not express a strong sentiment.")

        st.divider()

        st.write("### Input Text")

        st.write(user_input)