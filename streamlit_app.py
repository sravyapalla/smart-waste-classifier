import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2

# ── CONFIG ───────────────────────────────────────────────────
IMG_SIZE = (224, 224)
LABELS   = ['organic', 'recyclable', 'hazardous']

# ── LOAD MODEL ───────────────────────────────────────────────
# Make sure this file is committed to your repo alongside this script:

MODEL_PATH = "waste_classifier_final.keras"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ── UI LAYOUT ─────────────────────────────────────────────────
st.set_page_config(page_title="Waste Classifier", layout="centered")
st.title("♻️ Waste Classification Demo")
st.write("Upload a photo of trash and see which bin it belongs in.")

uploaded = st.file_uploader("Choose an image…", type=["jpg", "png", "jpeg"])
if uploaded:
    # Display the image
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Your upload", use_container_width=True)
    st.write("---")

    # Preprocess
    img_resized = img.resize(IMG_SIZE)
    x = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    # Predict
    probs = model.predict(x)[0]
    idx   = np.argmax(probs)
    label = LABELS[idx]
    conf  = probs[idx]

    # Show result
    st.markdown(f"**Prediction:** {label}")
    st.markdown(f"**Confidence:** {conf:.1%}")
