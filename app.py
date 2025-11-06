import os
import json
import numpy as np
import streamlit as st
from io import BytesIO
from PIL import Image

# -----------------------------------
# CONFIG
# -----------------------------------
st.set_page_config(page_title="CIFAR-10 Image Classifier", page_icon="🧠", layout="centered")

MODEL_PATH = "models/cifar10_cnn.h5"
LABELS_PATH = "models/class_names.json"
IMG_SIZE = (32, 32)

DEFAULT_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# -----------------------------------
# LOAD MODEL (cached)
# -----------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path):
    import tensorflow as tf
    from tensorflow import keras
    model = keras.models.load_model(model_path)
    return model

# -----------------------------------
# LOAD CLASS NAMES
# -----------------------------------
def load_class_names(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return DEFAULT_CLASSES

class_names = load_class_names(LABELS_PATH)

# -----------------------------------
# PREPROCESS FUNCTION
# -----------------------------------
def preprocess_image(img: Image.Image):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.asarray(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------------------
# SIDEBAR INFO
# -----------------------------------
with st.sidebar:
    st.header("Model Info")
    if os.path.exists(MODEL_PATH):
        st.success("Model loaded successfully ✅")
    else:
        st.error("Model file not found in models/")
        st.stop()

    st.write("Classes:")
    st.code(", ".join(class_names), language="text")

# -----------------------------------
# MAIN UI
# -----------------------------------
st.title("🧠 CIFAR-10 Image Classifier")
st.write("Upload an image, and the model will predict its class.")

uploaded_file = st.file_uploader("Upload an image (PNG or JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display image
    image = Image.open(BytesIO(uploaded_file.read()))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Load model
    model = load_model(MODEL_PATH)

    # Predict
    img_array = preprocess_image(image)
    prediction = model.predict(img_array, verbose=0)
    pred_class = class_names[int(np.argmax(prediction))]

    # Show result
    st.subheader("Predicted Class:")
    st.success(pred_class)

# Footer
st.markdown("---")
st.caption("Model trained on CIFAR-10 dataset • TensorFlow/Keras • Prediction based on uploaded image")
