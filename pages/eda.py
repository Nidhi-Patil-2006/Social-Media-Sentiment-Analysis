import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from wordcloud import WordCloud

# --------------------------------------------------
# Page Title
# --------------------------------------------------

st.title("📊 Exploratory Data Analysis")

st.markdown("Analysis of the cleaned social media sentiment dataset.")

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("dataset/cleaned_dataset.csv")

df = load_data()

# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------

st.header("Dataset Preview")

st.dataframe(df.head(10), use_container_width=True)

# --------------------------------------------------
# Dataset Information
# --------------------------------------------------

st.header("Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric("Missing Values", df.isnull().sum().sum())

# --------------------------------------------------
# Sentiment Distribution
# --------------------------------------------------

st.header("Sentiment Distribution")

counts = df["Sentiment"].value_counts()

col1, col2 = st.columns(2)

# Pie Chart

with col1:

    fig, ax = plt.subplots(figsize=(5,5))

    ax.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90
    )

    ax.set_title("Sentiment Distribution")

    st.pyplot(fig)

# Bar Chart

with col2:

    fig, ax = plt.subplots(figsize=(6,5))

    ax.bar(
        counts.index,
        counts.values
    )

    ax.set_title("Sentiment Count")

    ax.set_ylabel("Number of Posts")

    st.pyplot(fig)

# --------------------------------------------------
# Top 20 Words
# --------------------------------------------------

st.header("Most Frequent Words")

all_words = " ".join(df["Text"]).split()

word_counts = Counter(all_words)

top_words = word_counts.most_common(20)

words = [w[0] for w in top_words]

freq = [w[1] for w in top_words]

fig, ax = plt.subplots(figsize=(10,5))

ax.bar(words, freq)

ax.set_xticklabels(words, rotation=45, ha="right")

ax.set_title("Top 20 Most Frequent Words")

st.pyplot(fig)

# --------------------------------------------------
# Sentence Length Distribution
# --------------------------------------------------

st.header("Sentence Length Distribution")

lengths = df["Text"].apply(lambda x: len(x.split()))

fig, ax = plt.subplots(figsize=(8,5))

ax.hist(lengths, bins=20)

ax.set_xlabel("Number of Words")

ax.set_ylabel("Frequency")

ax.set_title("Sentence Length Distribution")

st.pyplot(fig)

# --------------------------------------------------
# Word Clouds
# --------------------------------------------------

st.header("Word Clouds")

for sentiment in ["Positive", "Neutral", "Negative"]:

    st.subheader(f"{sentiment} Sentiment")

    text = " ".join(
        df[df["Sentiment"] == sentiment]["Text"]
    )

    wordcloud = WordCloud(
        width=900,
        height=450,
        background_color="white"
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10,5))

    ax.imshow(wordcloud)

    ax.axis("off")

    st.pyplot(fig)

# --------------------------------------------------
# Class Counts
# --------------------------------------------------

st.header("Class Counts")

st.dataframe(
    counts.reset_index().rename(
        columns={
            "index":"Sentiment",
            "Sentiment":"Count"
        }
    ),
    use_container_width=True
)