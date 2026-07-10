import re
import string
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --------------------------------------------------
# Download required NLTK resources
# --------------------------------------------------
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# --------------------------------------------------
# Initialize NLP tools
# --------------------------------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# --------------------------------------------------
# Sentiment Mapping
# --------------------------------------------------
SENTIMENT_MAPPING = {

    # -------------------------
    # Positive
    # -------------------------
    "Positive": "Positive",
    "Acceptance": "Positive",
    "Accomplishment": "Positive",
    "Admiration": "Positive",
    "Adoration": "Positive",
    "Adrenaline": "Positive",
    "Adventure": "Positive",
    "Affection": "Positive",
    "Amazement": "Positive",
    "Amusement": "Positive",
    "Anticipation": "Positive",
    "Appreciation": "Positive",
    "Arousal": "Positive",
    "ArtisticBurst": "Positive",
    "Awe": "Positive",
    "Blessed": "Positive",
    "Breakthrough": "Positive",
    "Captivation": "Positive",
    "Celebration": "Positive",
    "Celestial Wonder": "Positive",
    "Challenge": "Positive",
    "Charm": "Positive",
    "Colorful": "Positive",
    "Compassion": "Positive",
    "Compassionate": "Positive",
    "Confidence": "Positive",
    "Confident": "Positive",
    "Connection": "Positive",
    "Contentment": "Positive",
    "Coziness": "Positive",
    "Creative Inspiration": "Positive",
    "Creativity": "Positive",
    "Culinary Adventure": "Positive",
    "CulinaryOdyssey": "Positive",
    "Dazzle": "Positive",
    "Determination": "Positive",
    "DreamChaser": "Positive",
    "Ecstasy": "Positive",
    "Elation": "Positive",
    "Elegance": "Positive",
    "Empathetic": "Positive",
    "Empowerment": "Positive",
    "Enchantment": "Positive",
    "Energy": "Positive",
    "Engagement": "Positive",
    "Enjoyment": "Positive",
    "Enthusiasm": "Positive",
    "Envisioning History": "Positive",
    "Euphoria": "Positive",
    "Excitement": "Positive",
    "Exploration": "Positive",
    "FestiveJoy": "Positive",
    "Free-spirited": "Positive",
    "Freedom": "Positive",
    "Friendship": "Positive",
    "Fulfillment": "Positive",
    "Grandeur": "Positive",
    "Grateful": "Positive",
    "Gratitude": "Positive",
    "Happiness": "Positive",
    "Happy": "Positive",
    "Harmony": "Positive",
    "Heartwarming": "Positive",
    "Hope": "Positive",
    "Hopeful": "Positive",
    "Hypnotic": "Positive",
    "Iconic": "Positive",
    "Imagination": "Positive",
    "Inspiration": "Positive",
    "Inspired": "Positive",
    "Intrigue": "Positive",
    "Journey": "Positive",
    "Joy": "Positive",
    "Joy in Baking": "Positive",
    "JoyfulReunion": "Positive",
    "Kind": "Positive",
    "Kindness": "Positive",
    "Love": "Positive",
    "Marvel": "Positive",
    "Melodic": "Positive",
    "Mesmerizing": "Positive",
    "Mindfulness": "Positive",
    "Motivation": "Positive",
    "Nature's Beauty": "Positive",
    "Optimism": "Positive",
    "Overjoyed": "Positive",
    "Playful": "Positive",
    "PlayfulJoy": "Positive",
    "Positivity": "Positive",
    "Pride": "Positive",
    "Proud": "Positive",
    "Radiance": "Positive",
    "Rejuvenation": "Positive",
    "Relief": "Positive",
    "Renewed Effort": "Positive",
    "Resilience": "Positive",
    "Reverence": "Positive",
    "Romance": "Positive",
    "Satisfaction": "Positive",
    "Spark": "Positive",
    "Success": "Positive",
    "Surprise": "Positive",
    "Tenderness": "Positive",
    "Thrill": "Positive",
    "Thrilling Journey": "Positive",
    "Touched": "Positive",
    "Triumph": "Positive",
    "Vibrancy": "Positive",
    "Whimsy": "Positive",
    "Winter Magic": "Positive",
    "Wonder": "Positive",
    "Wonderment": "Positive",
    "Zest": "Positive",

    # -------------------------
    # Neutral
    # -------------------------
    "Neutral": "Neutral",
    "Ambivalence": "Neutral",
    "Calmness": "Neutral",
    "Confusion": "Neutral",
    "Contemplation": "Neutral",
    "Curiosity": "Neutral",
    "Emotion": "Neutral",
    "Indifference": "Neutral",
    "InnerJourney": "Neutral",
    "Miscalculation": "Neutral",
    "Nostalgia": "Neutral",
    "Pensive": "Neutral",
    "Reflection": "Neutral",
    "Serenity": "Neutral",
    "Solace": "Neutral",
    "Solitude": "Neutral",
    "Suspense": "Neutral",
    "Tranquility": "Neutral",

    # -------------------------
    # Negative
    # -------------------------
    "Negative": "Negative",
    "Anger": "Negative",
    "Anxiety": "Negative",
    "Apprehensive": "Negative",
    "Bad": "Negative",
    "Betrayal": "Negative",
    "Bitter": "Negative",
    "Bitterness": "Negative",
    "Boredom": "Negative",
    "Darkness": "Negative",
    "Desolation": "Negative",
    "Despair": "Negative",
    "Desperation": "Negative",
    "Devastated": "Negative",
    "Disappointed": "Negative",
    "Disappointment": "Negative",
    "Disgust": "Negative",
    "Dismissive": "Negative",
    "Embarrassed": "Negative",
    "EmotionalStorm": "Negative",
    "Envious": "Negative",
    "Envy": "Negative",
    "Exhaustion": "Negative",
    "Fear": "Negative",
    "Fearful": "Negative",
    "Frustrated": "Negative",
    "Frustration": "Negative",
    "Grief": "Negative",
    "Hate": "Negative",
    "Heartache": "Negative",
    "Heartbreak": "Negative",
    "Helplessness": "Negative",
    "Intimidation": "Negative",
    "Isolation": "Negative",
    "Jealous": "Negative",
    "Jealousy": "Negative",
    "Loneliness": "Negative",
    "Loss": "Negative",
    "LostLove": "Negative",
    "Melancholy": "Negative",
    "Numbness": "Negative",
    "Obstacle": "Negative",
    "Overwhelmed": "Negative",
    "Pressure": "Negative",
    "Regret": "Negative",
    "Resentment": "Negative",
    "Ruins": "Negative",
    "Sad": "Negative",
    "Sadness": "Negative",
    "Shame": "Negative",
    "Sorrow": "Negative",
    "Suffering": "Negative",
    "Sympathy": "Negative",
    "Whispers of the Past": "Negative",
    "Yearning": "Negative"
}

# --------------------------------------------------
# Text Cleaning Function
# --------------------------------------------------
def clean_text(text):

    if pd.isna(text):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtag symbol
    text = text.replace("#", "")

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # remove emojis
    text = text.encode("ascii", "ignore").decode()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords and lemmatize
    cleaned_words = []

    for word in text.split():

        if word not in stop_words:

            word = lemmatizer.lemmatize(word)

            cleaned_words.append(word)

    return " ".join(cleaned_words)


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
def load_dataset(path):

    return pd.read_csv(path)


# --------------------------------------------------
# Preprocess Dataset
# --------------------------------------------------
def preprocess_dataframe(df):

    print("\nOriginal Dataset Shape:", df.shape)

    # Remove unnecessary columns
    df = df.drop(
        columns=["Unnamed: 0", "Unnamed: 0.1"],
        errors="ignore"
    )

    # Keep only required columns
    df = df[["Text", "Sentiment"]]

    # Remove missing values
    df = df.dropna()

    # Remove leading/trailing spaces from ALL text columns
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].astype(str).str.strip()

    # Print original labels
    print("\nOriginal Labels:")
    print(sorted(df["Sentiment"].unique()))

    # Map sentiments
    df["Sentiment"] = df["Sentiment"].map(SENTIMENT_MAPPING)

    # Show mapping result
    print("\nSentiment Distribution After Mapping:")
    print(df["Sentiment"].value_counts(dropna=False))

    # Remove unmapped rows
    df = df.dropna(subset=["Sentiment"])

    print("\nCleaning text...")

    # Clean text
    df["Text"] = df["Text"].apply(clean_text)

    # Remove empty texts
    df = df[df["Text"].str.strip() != ""]

    print("\nFinal Dataset Shape:", df.shape)

    return df


# --------------------------------------------------
# Save Dataset
# --------------------------------------------------
def save_dataset(df, path):

    df.to_csv(path, index=False)

    print(f"\nCleaned dataset saved successfully!")
    print(f"Location: {path}")


# --------------------------------------------------
# Main Function
# --------------------------------------------------
def main():

    INPUT_PATH = "dataset/raw/sentimentdataset.csv"
    OUTPUT_PATH = "dataset/cleaned_dataset.csv"

    print("Loading dataset...")

    df = load_dataset(INPUT_PATH)

    df = preprocess_dataframe(df)

    print("\nFinal Sentiment Distribution:")
    print(df["Sentiment"].value_counts())

    print("\nSample Data:")
    print(df.head())

    save_dataset(df, OUTPUT_PATH)

    print("\nPreprocessing Completed Successfully!")


# --------------------------------------------------
# Run Program
# --------------------------------------------------
if __name__ == "__main__":
    main()