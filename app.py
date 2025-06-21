
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- SETTINGS ---
MODEL_PATH = "mobilenetv2_banknote_predictor.h5"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["50", "100", "200", "500", "1000", "5000", "10000"]

# --- Load Model ---
@st.cache_resource
def load_keras_model():
    model = load_model(MODEL_PATH)
    return model

model = load_keras_model()

# --- Preprocess Function ---
def preprocess_image(image: Image.Image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image.convert("RGB"))  # Ensure 3 channels
    img_array = preprocess_input(img_array)     # MobileNetV2 preprocessing
    return np.expand_dims(img_array, axis=0)    # (1, 224, 224, 3)

# --- Streamlit App ---
st.set_page_config(page_title="Banknote Classifier", layout="centered")
st.title("💵 Myanmar Banknote Recognition (MobileNetV2)")
st.caption("Upload an image of a Myanmar banknote to identify its denomination.")

uploaded_file = st.file_uploader("📤 Upload a banknote image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Banknote", use_column_width=True)

    with st.spinner("🔍 Predicting..."):
        input_tensor = preprocess_image(image)
        prediction = model.predict(input_tensor)[0]

        pred_class_index = np.argmax(prediction)
        pred_class_label = CLASS_NAMES[pred_class_index]
        confidence = prediction[pred_class_index]

    st.success(f"✅ Predicted Denomination: **{pred_class_label} Ks**")
    st.info(f"📊 Confidence: {confidence * 100:.2f}%")

    # Show all class probabilities
    st.markdown("### 🔢 Class Probabilities:")
    for i, prob in enumerate(prediction):
        st.write(f"- {CLASS_NAMES[i]} Ks: {prob * 100:.2f}%")
