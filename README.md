# 📰 Fake News Detection using Deep Learning

A real-time, interactive web application that detects fake news using a deep learning model and natural language processing techniques. Users can input any news article or headline and instantly find out whether it’s likely to be **REAL** or **FAKE**, along with **sentiment analysis**, **visual confidence plots**, and the option to export prediction history.

---

## 📘 Dataset
Kaggle Dataset: [Fake News Detection Dataset](https://www.kaggle.com/datasets/jruvika/fake-news-detection)

Contains labeled examples of real and fake news for supervised learning.

---

## 🚀 Features

- 🔍 Real-time fake news classification with confidence scores
- 😊 Sentiment analysis of the news content
- 📈 Interactive visualization: Confidence vs. Sentiment scatter plot
- 📸 Downloadable plot image and session data (CSV)
- 📊 Histogram of model confidence (optional analytics)
- 💾 Built with TensorFlow, Scikit-learn, Streamlit, and Plotly

---

## 📂 Project Structure

├── app-fakenews.py # Main Streamlit app 

├── fake_news_model.keras # Trained deep learning model 

├── vectorizer.pkl # Pickled text vectorizer 

├── requirements.txt # Dependencies 

└── README.md # You're here!


---
```yaml
🧠 Model Overview

- Model Type: Binary Classifier (Real vs Fake)
- Architecture: Deep Neural Network (can be extended to BERT or LSTM)
- Input: News headline or article
- Preprocessing: Tokenization, stopword removal, stemming
- Vectorization: TF-IDF or custom tokenizer
- Output: Binary class + probability score
```
---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
```

2. **Create a virtual environment and install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the app**

```bash
streamlit run app-fakenews.py
```

---

## 🛠️ To-Do / Future Enhancements
- [ ]  Fine-tune a transformer model (e.g., BERT)

- [ ] Explain predictions with LIME or SHAP

- [ ] Add support for CSV or RSS feed inputs

- [ ] Deploy via Hugging Face Spaces or Streamlit Cloud

- [ ] Add multilingual support with XLM-R

---

## 🧪 Example Usage
1. Paste a news article or headline into the text box.

2. Click Analyze to get:

    - Real/Fake classification

    - Confidence score

    - Sentiment analysis

3. Click Show Visual to:

    - Plot confidence vs sentiment

    - Export the results as CSV or PNG


---

## 🧑‍💻 Authors

Alekhya Vittalam
Ananya Agrawal
Ananya Chembai
Chehak Arora
Sophia Kurz



