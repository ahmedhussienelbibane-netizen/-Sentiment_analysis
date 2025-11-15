import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
import numpy as np
import os

st.set_page_config(page_title="Sentiment Classifier", layout="wide")
st.title("💬 Sentiment Classification App")

# ========================
# Load tokenizer from HuggingFace
# ========================
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

label_map = {0: "Negative", 1: "Neutral", 2: "Positive", 3: "Irrelevant"}

# ========================
# Rebuild model architecture
# ========================
def build_model(num_labels=4):
    base_model = TFAutoModel.from_pretrained(MODEL_NAME)

    input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")

    outputs = base_model(input_ids, attention_mask=attention_mask)[0]
    cls_token = outputs[:, 0, :]  # CLS token embedding

    logits = tf.keras.layers.Dense(num_labels, activation=None)(cls_token)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=logits)
    return model

# ========================
# Build model + load weights automatically
# ========================
model = build_model()
weights_file = "weights_only.keras"

if not os.path.exists(weights_file):
    st.error(f"Error: weights file '{weights_file}' not found in the same folder as the app.")
else:
    model.load_weights(weights_file)
    st.success("Model loaded successfully!")

# ========================
# Prediction function
# ========================
def predict(text):
    enc = tokenizer(
        text,
        return_tensors="tf",
        truncation=True,
        padding=True,
        max_length=64
    )
    logits = model([enc["input_ids"], enc["attention_mask"]])
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
    pred = int(np.argmax(probs))
    return label_map[pred], probs

# ========================
# Streamlit UI
# ========================
user_text = st.text_area("Enter text:", height=150)

if st.button("Analyze"):
    if not user_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            prediction, probabilities = predict(user_text)
        
        st.subheader("Prediction:")
        st.success(prediction)
        
        st.subheader("Confidence:")
        for i, p in enumerate(probabilities):
            st.write(f"**{label_map[i]}:** {round(float(p)*100, 2)}%")
