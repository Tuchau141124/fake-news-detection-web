import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# Load mô hình
MODEL_PATH = '/Users/tuchou/Downloads/TTNT/ AI/fake_news_detection_model.h5'  # hoặc 'fake_news_model.joblib'
TOKENIZER_PATH = '/Users/tuchou/Downloads/TTNT/ AI/tokenizer.json'

# Tải mô hình
@st.cache_resource  # Cache để tăng tốc độ tải
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# Tải tokenizer
@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, 'r') as f:
        return tokenizer_from_json(json.load(f))

# Tiền xử lý dữ liệu
def preprocess_text(text, tokenizer, max_len=300):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded

# Dự đoán
def predict_news(text, model, tokenizer):
    processed_text = preprocess_text(text, tokenizer)
    prediction = model.predict(processed_text)[0][0]
    return "Fake News" if prediction > 0.5 else "Real News"

# Streamlit UI
st.title("Fake News Detection App")
st.write("Check if a news article is real or fake.")

# Nhập văn bản
user_input = st.text_area("Paste your news article here:")

if st.button("Check"):
    if user_input.strip():
        # Load mô hình và tokenizer
        model = load_model()
        tokenizer = load_tokenizer()

        # Thực hiện dự đoán
        result = predict_news(user_input, model, tokenizer)
        st.write(f"The article is likely: **{result}**")
    else:
        st.write("Please enter some text.")