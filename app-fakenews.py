import streamlit as st
import pickle
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import tensorflow as tf
from textblob import TextBlob
import plotly.express as px

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Styling
st.set_page_config(page_title="📰 Fake News Detector", layout="centered", page_icon="🧠")
ps = PorterStemmer()

# Apply styles
def apply_styles():
    st.markdown("""
        <style>
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

# Load model and vectorizer
@st.cache_resource
def load_components():
    try:
        model = tf.keras.models.load_model("fake_news_model.keras", compile=False)
        model.compile()
        with open("vectorization.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        st.error(f"❌ Failed to load model/vectorizer: {e}")
        return None, None

model, vectorizer = load_components()

# Preprocessing
def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

# Prediction
def predict_news(news_text):
    if model is None or vectorizer is None:
        st.error("Model or vectorizer not loaded.")
        return None
    cleaned = preprocess(news_text)
    vectorized_input = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(vectorized_input)[0][0]
    return prediction

# Sentiment
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, polarity

# Initialize session state
if "plot_data" not in st.session_state:
    st.session_state.plot_data = []

# UI Container
with st.container():
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.title("📰 Fake News Detector")
    st.markdown("Paste a news headline or article below to check if it's **real or fake**, and get the **sentiment** of the news.")

    news_input = st.text_area("🖊️ Your News Text:", height=200, placeholder="Enter the news headline or article...")

    if st.button("🔍 Analyze"):
        if not news_input.strip():
            st.warning("⚠️ Please enter some news content.")
        else:
            with st.spinner("Analyzing..."):
                prediction_score = predict_news(news_input)
                sentiment_label, polarity_score = analyze_sentiment(news_input)

            if prediction_score is None:
                st.error("❌ Unable to analyze. Try again later.")
            else:
                is_real = prediction_score > 0.5
                confidence = prediction_score if is_real else 1 - prediction_score

                if is_real:
                    st.success(f"✅ This news appears to be **REAL** (Confidence: {confidence*100:.2f}%)")
                else:
                    st.error(f"🚨 This news appears to be **FAKE** (Confidence: {confidence*100:.2f}%)")

                sentiment_emojis = {"Positive": "😊", "Negative": "😠", "Neutral": "😐"}
                emoji = sentiment_emojis.get(sentiment_label, "💬")

                st.markdown(f"""
                ---
                ### {emoji} Sentiment Analysis
                - **Sentiment:** `{sentiment_label}`
                - **Polarity Score:** `{polarity_score:.2f}`
                """)

    if st.button("📊 Show Visual"):
        if not news_input.strip():
            st.warning("⚠️ Enter news to visualize.")
        else:
            prediction_score = predict_news(news_input)
            sentiment_label, polarity_score = analyze_sentiment(news_input)
            confidence = prediction_score if prediction_score > 0.5 else 1 - prediction_score
            label = "Real" if prediction_score > 0.5 else "Fake"

            st.session_state.plot_data.append({
                "text": news_input[:200],
                "confidence": confidence,
                "sentiment": polarity_score,
                "label": label
            })

            df_viz = pd.DataFrame(st.session_state.plot_data)

            fig = px.scatter(
                df_viz,
                x="confidence",
                y="sentiment",
                color="label",
                hover_data=["text"],
                labels={"confidence": "Confidence", "sentiment": "Sentiment Polarity"},
                title="🧠 Confidence vs Sentiment",
                color_discrete_map={"Real": "green", "Fake": "red"}
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
