import streamlit as st
import pickle
import re
import string

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text cleaning (same as training)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# UI
st.title("📰 Fake News Detection App")

input_text = st.text_area("Enter News Text:")

if st.button("Predict"):
    if input_text:
        cleaned = clean_text(input_text)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.success("✅ Real News")
        else:
            st.error("❌ Fake News")
    else:
        st.warning("Please enter some text")
