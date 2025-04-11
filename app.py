import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import tensorflow as tf

# Page configuration
st.set_page_config(page_title="📰 Fake News Detector", layout="centered", page_icon="🧠")

# Download stopwords
nltk.download('stopwords')
ps = PorterStemmer()

# Load model and vectorizer with caching
@st.cache_resource
def load_components():
    try:
        model = tf.keras.models.load_model("fake_news_model.keras", compile=False)
        model.compile()
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        st.error(f"Failed to load model/vectorizer: {e}")
        return None, None

model, vectorizer = load_components()

# Preprocessing function
def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

# Prediction function
def predict_news(news_text):
    if model is None or vectorizer is None:
        st.error("Model or vectorizer not loaded.")
        return None
    cleaned = preprocess(news_text)
    vectorized_input = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(vectorized_input)[0][0]
    return prediction > 0.5

# Apply custom background and styles
def apply_styles():
    st.markdown("""
        <style>
        body {
            font-family: 'Segoe UI', sans-serif;
        }
        .stApp {
            background-image: linear-gradient(to bottom right, #f5f7fa, #c3cfe2);
            background-attachment: fixed;
            background-size: cover;
            color: #222;
        }
        .main-card {
            background: white;
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            max-width: 700px;
            margin: 2rem auto;
        }
        .stTextArea textarea {
            font-size: 1rem;
            line-height: 1.5;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #4CAF50 !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
            transition: 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #45a049 !important;
            transform: scale(1.03);
        }
        </style>
    """, unsafe_allow_html=True)

apply_styles()

# Main App Layout
with st.container():
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.title("📰 Fake News Detector")
    st.markdown("Paste a news headline or article below to check if it's **real or fake**.")

    news_input = st.text_area("🖊️ Your News Text:", height=200, placeholder="Enter the news headline or article...")

    if st.button("🔍 Analyze"):
        if not news_input.strip():
            st.warning("⚠️ Please enter some news content.")
        else:
            with st.spinner("Analyzing..."):
                result = predict_news(news_input)
            if result is None:
                st.error("❌ Unable to analyze. Try again later.")
            elif result:
                st.success("✅ This news appears to be **REAL**.")
            else:
                st.error("🚨 This news appears to be **FAKE**.")
    st.markdown("</div>", unsafe_allow_html=True)
