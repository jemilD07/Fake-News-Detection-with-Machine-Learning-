# app.py
import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load("logistic_regression_model.pkl")  # or naive_bayes_model.pkl
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit UI
st.set_page_config(page_title="📰 Fake News Detector", layout="centered")
st.title("🧠 Fake News Detection App")
st.markdown(
    "Enter a news article and the model will predict if it's **Fake** or **Real**."
)

# Input text
user_input = st.text_area("✍️ Paste News Article Text Here", height=250)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # Preprocess and transform the text
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        label = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
        st.subheader("Prediction:")
        st.markdown(f"## {label}")
