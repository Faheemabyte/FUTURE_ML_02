import string
import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

@st.cache_resource
def setup():
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")
    vec = joblib.load("tfidf_vectorizer.pkl")
    cat = joblib.load("ticket_classifier_model.pkl")
    pri = joblib.load("ticket_priority_model.pkl")
    return set(stopwords.words("english")), vec, cat, pri

stop_words, vec, cat_model, pri_model = setup()

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

st.title("Support Ticket Classifier")
st.write("Paste a customer support ticket and the model predicts its category and priority.")

ticket = st.text_area("Ticket text", height=150)

if st.button("Classify") and ticket.strip():
    X = vec.transform([clean_text(ticket)])
    st.success(f"Category: {cat_model.predict(X)[0]}")
    st.info(f"Priority: {pri_model.predict(X)[0]}")
