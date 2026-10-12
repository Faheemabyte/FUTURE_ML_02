import joblib
import streamlit as st

@st.cache_resource
def load_models():
    vec = joblib.load("tfidf_vectorizer (1).pkl")
    cat = joblib.load("ticket_classifier_model.pkl")
    pri = joblib.load("ticket_priority_model (1).pkl")
    return vec, cat, pri

vec, cat_model, pri_model = load_models()

st.title("Support Ticket Classifier")
st.write("Paste a customer support ticket and the model predicts its category and priority.")

ticket = st.text_area("Ticket text", height=150)

if st.button("Classify") and ticket.strip():
    X = vec.transform([ticket])
    st.success(f"Category: {cat_model.predict(X)[0]}")
    st.info(f"Priority: {pri_model.predict(X)[0]}")
