
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
if st.button("Predict"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Convert text into numerical features
        text_vector = vectorizer.transform([user_text])

        # Predict the category
        prediction = model.predict(text_vector)

        st.success(f"Predicted category: {prediction[0]}")

        # Display probabilities if supported
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(text_vector)[0]

            st.subheader("Prediction Probabilities")

            for label, probability in zip(
                model.classes_, probabilities
            ):
                st.write(f"{label}: {probability:.2%}")
