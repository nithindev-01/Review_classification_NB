
import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Page settings
st.set_page_config(
    page_title="Text Classification App",
    page_icon="🤖"
)

st.title("🤖 Text Classification using Naive Bayes")
st.write("Enter some text and let the model predict its category.")

# User input
user_text = st.text_area("Enter your text here:")

# Prediction button


if st.button("Predict Sentiment"):
    if user_text.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned_text = clean_text(
            user_text, stop_words, lemmatizer
        )

        text_vector = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_vector)[0]

        if prediction == 1:
            st.success("Prediction: Positive 😊")
        elif prediction == 0:
            st.error("Prediction: Negative ☹️")
        else:
            st.write("Unexpected prediction:", prediction)
