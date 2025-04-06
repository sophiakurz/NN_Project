import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import tensorflow as tf
import streamlit as st

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")


# Initialize NLTK
nltk.download('stopwords')
ps = PorterStemmer()

# Load the model and vectorizer
@st.cache_resource
def load_components():
    try:
        model = tf.keras.models.load_model(
            "fake_news_model.keras",
            compile=False
        )
        model.compile()  
        
        # Load vectorizer
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
            
        return model, vectorizer
    except Exception as e:
        st.error(f"Loading failed: {str(e)}")
        return None, None

model, vectorizer = load_components()


def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

def predict_news(news_text):
    if model is None or vectorizer is None:
        st.error("Model not loaded properly")
        return False
        
    cleaned = preprocess(news_text)
    vectorized_input = vectorizer.transform([cleaned]).toarray()  
    prediction = model.predict(vectorized_input)[0][0]
    return prediction > 0.5




# Streamlit App
# CSS
def set_bg_image():
    st.markdown(
        f"""
        <style>

        .stApp {{
            background-image: linear-gradient(rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.6)), 
            url("https://images.pexels.com/photos/27164019/pexels-photo-27164019/free-photo-of-dried-pink-rose-on-yellowed-newspaper.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2");
            background-attachment: fixed;
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            color: black;
        }}
        .stButton>button {{
            background-color: rgba(255, 255, 255, 0.8) !important;
            border-radius: 10px !important;
            padding: 10px 20px !important;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
            color: black !important;  /* Changed from white to black */
            border: 1px solid #ccc !important;
            font-weight: bold !important;
            transition: all 0.3s ease !important;
        }}
        /* Button hover effect */
        .stButton>button:hover {{
            background-color: rgba(240, 240, 240, 0.9) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2) !important;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_image()

st.title("📰 Fake News Detector")
st.markdown("Paste any news headline or paragraph and find out if it's **real or fake**.")

news_input = st.text_area("Enter the news text here:")

if st.button("Detect"):
    if not news_input.strip():
        st.warning("Please enter some news text first!")
    else:
        result = predict_news(news_input)
        if result:
            st.success("✅ This news is REAL.")
        else:
            st.error("🚨 This news is FAKE.")
