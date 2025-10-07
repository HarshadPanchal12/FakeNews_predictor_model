import streamlit as st
import pickle
import re

# Load model and vectorizer
model = pickle.load(open("logistic_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+|[^a-zA-Z\s]", "", text)
    return text.strip()

# Streamlit App UI
st.title("📰 Fake News Headline Detector")
st.write("Enter a news headline to check if it's fake or real.")

user_input = st.text_area("📝 Headline", "")

if st.button("Predict"):
    if user_input:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned]).toarray()
        prediction = model.predict(vec)[0]

        if prediction == 1:
            st.error("🚨 This is likely a FAKE news headline!")
        else:
            st.success("✅ This seems to be a REAL news headline.")
    else:
        st.warning("Please enter a headline to analyze.")
