import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# Load tokenizer
with open('/Users/tuchou/Downloads/Final Quản trị dự án CNTT/ AI/tokenizer.json', 'r') as f:
    tokenizer_data = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_data)

# Load model
model = tf.keras.models.load_model('fake_news_model.h5')

# Function to predict
def predict_news(text):
    # Tokenize and pad the input text
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=300)
    
    # Predict with the model
    prediction = model.predict(padded_sequence)
    return "Fake News" if prediction[0] > 0.5 else "Real News"

# Streamlit UI
st.title("Fake News Detection")
st.write("Enter a news article to check if it's real or fake.")

# Input box
user_input = st.text_area("Paste your news article here:")

if st.button("Check News"):
    if user_input:
        result = predict_news(user_input)
        st.write(f"The article is likely: **{result}**")
    else:
        st.write("Please enter some text.")