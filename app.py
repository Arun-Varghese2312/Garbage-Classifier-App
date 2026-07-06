import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import os

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Garbage Classifier", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "garbage_classifier.keras")
CLASS_PATH = os.path.join(BASE_DIR, "class_names.json")

IMG_SIZE = (180, 180)

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

model = load_model()

# ----------------------------
# LOAD CLASS NAMES
# ----------------------------
with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

# ----------------------------
# IMAGE PREPROCESSING
# ----------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image, dtype=np.float32)


    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# ----------------------------
# UI
# ----------------------------
st.title("🗑️ Garbage Classification App")
st.write("Upload an image to classify the type of waste.")

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = preprocess_image(image)

    prediction = model.predict(img, verbose=0)

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = prediction[0][predicted_index]

    st.subheader("Prediction")

    st.success(f"{predicted_class}")

    st.write(f"Confidence: {confidence*100:.2f}%")

    st.subheader("Top 5 Predictions")

    top5 = np.argsort(prediction[0])[::-1][:5]

    for idx in top5:
        st.write(f"{class_names[idx]} : {prediction[0][idx]*100:.2f}%")