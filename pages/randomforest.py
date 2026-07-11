import streamlit as st

from src.predict import predict_random_forest

# --------------------------------------------------
# Page Title
# --------------------------------------------------

st.title("🌳 Random Forest Predictor")


st.markdown("""
### About the Model

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting. In this project, it classifies social media posts into **Positive**, **Negative**, or **Neutral** sentiments.

#### Features
- ✅ Ensemble Learning Algorithm
- ✅ 500 Decision Trees
- ✅ Tuned Hyperparameters
- ✅ Balanced Class Weights
""")

st.divider()
st.markdown("""
Enter a social media post below to predict its sentiment using the
**Random Forest** model.
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

        prediction = predict_random_forest(user_input)

        st.subheader("Prediction")

        if prediction == "Positive":

            st.success(
                "🟢 **Positive**\n\n"
                "The post expresses a favourable or optimistic sentiment."
            )

        elif prediction == "Negative":

            st.error(
                "🔴 **Negative**\n\n"
                "The post expresses an unfavourable or negative sentiment."
            )

        else:

            st.info(
                "⚪ **Neutral**\n\n"
                "The post is informational or does not express a strong sentiment."
            )

        st.divider()

        st.subheader("Input Text")

        st.write(user_input)