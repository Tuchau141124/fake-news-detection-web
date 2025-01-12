import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import os

# Kiểm tra sự tồn tại của các tệp quan trọng
MODEL_PATH = "/Users/tuchou/Downloads/TTNT/ AI/fake_news_detection_model.h5"
TOKENIZER_PATH = "/Users/tuchou/Downloads/TTNT/ AI/tokenizer.json"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
elif not os.path.exists(TOKENIZER_PATH):
    st.error(f"Tokenizer file not found: {TOKENIZER_PATH}")
else:
    # Load tokenizer
    with open(TOKENIZER_PATH, 'r') as f:
        tokenizer_data = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_data)

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Function to predict
    def predict_news(text):
        # Tokenize and pad the input text
        sequences = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequences, maxlen=300)
        
        # Predict with the model
        prediction = model.predict(padded_sequence)
        return "Fake News" if prediction[0] > 0.5 else "Real News"

    # Streamlit UI
    st.title("📰 Fake News Detection")
    st.write("This application helps you detect whether a news article is **Fake** or **Real**. Simply paste the article below!")

    # Input box
    user_input = st.text_area("Paste your news article here:", height=200)

    # Predict button
    if st.button("Analyze"):
        if not user_input.strip():
            st.warning("Please enter some text to analyze!")
        else:
            try:
                result = predict_news(user_input)
                st.success(f"The article is likely: **{result}**")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Footer
    st.markdown("---")
    st.markdown("Developed with ❤️ using Streamlit and TensorFlow.")
