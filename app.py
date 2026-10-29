
import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
@st.cache_resource
def load_nltk_resources():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    return (
        set(stopwords.words("english")),
        WordNetLemmatizer()
    )

stop_words, lemmatizer = load_nltk_resources()

# Preserve important words, as in the notebook
important_words = {
    "no", "not", "nor", "neither", "never",
    "nothing", "nowhere", "hardly", "barely", "scarcely",
    "but", "however", "although", "though", "yet", "while",
    "very", "too", "so", "really", "quite",
    "extremely", "highly", "completely", "absolutely"
}

stop_words -= important_words

# Load the single PKL file
@st.cache_resource
def load_model():
    return joblib.load("naive_bayes_model.pkl")

bundle = load_model()
model = bundle["model"]
vectorizer = bundle["vectorizer"]

# Text preprocessing from your notebook
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Streamlit interface
st.title("Review Sentiment Analysis")
st.write("Enter a review to predict whether its sentiment is positive or negative.")

review = st.text_area("Enter your review:")

if st.button("Predict Sentiment"):
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        cleaned_review = clean_text(review)

        if not cleaned_review:
            st.warning("The review has no usable words after preprocessing.")
        else:
            review_tfidf = vectorizer.transform([cleaned_review])
            prediction = model.predict(review_tfidf)[0]

            if prediction == 1:
                st.success("Positive Sentiment 😊")
            else:
                st.error("Negative Sentiment 😞")
