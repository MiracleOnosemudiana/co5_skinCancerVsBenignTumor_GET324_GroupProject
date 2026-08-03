import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Skin Cancer vs Benign Tumor Classifier",
    layout="centered"
)


# LOAD MODEL

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/custom_cnn_best.keras")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model.\n\n{e}")
    st.stop()

IMAGE_SIZE = (128, 128)


# PREPROCESS

def preprocess(image):
    image = image.resize(IMAGE_SIZE)
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image



# PREDICTION

def predict(image):

    x = preprocess(image)

    prediction = float(model.predict(x, verbose=0)[0][0])

    if prediction >= 0.5:
        label = "Skin Cancer"
        confidence = prediction
    else:
        label = "Benign Tumor"
        confidence = 1 - prediction

    return label, confidence, prediction



# TITLE

st.title("Skin Cancer vs Benign Tumor Classification")

st.markdown("""
### Upload Instructions

- Upload a **clear dermoscopic image**
- One lesion per image
- Supported formats: **JPG, JPEG, PNG**
- Avoid blurry or edited images

The model predicts whether the lesion is:

**Benign Tumor** or **Skin Cancer**
""")

st.warning(
    "This application is for educational purposes only and "
    "must not be used for medical diagnosis."
)


# SAMPLE IMAGES

st.header("📷 Sample Images")

sample_images = {
    "Benign 1": "sample_data/benign_01.jpg",
    "Benign 2": "sample_data/benign_02.jpg",
    "Cancer 1": "sample_data/cancer_01.jpg",
    "Cancer 2": "sample_data/cancer_02.jpg",
}

cols = st.columns(2)

for i, (name, path) in enumerate(sample_images.items()):

    with cols[i % 2]:

        if os.path.exists(path):

            st.image(path, width=220)

            if st.button(f"Use {name}", key=name):
                st.session_state["selected_image"] = path
                st.rerun()

        else:
            st.warning(f"Missing:\n{path}")


# UPLOAD

st.header("Upload Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.session_state.pop("selected_image", None)


# DETERMINE IMAGE

image = None

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

elif "selected_image" in st.session_state:

    image = Image.open(
        st.session_state["selected_image"]
    ).convert("RGB")


# RUN PREDICTION

if image is not None:

    st.divider()

    st.subheader("Selected Image")

    c1, c2, c3 = st.columns([1,2,1])

    with c2:
        st.image(image, width=300)

    if st.button("Predict"):

        with st.spinner("Analyzing image..."):

            label, confidence, raw = predict(image)

        st.divider()

        st.subheader("Prediction")

        if label == "Skin Cancer":
            st.error(label)
        else:
            st.success(label)

        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

        st.progress(confidence)

        st.write(f"Probability of Skin Cancer: **{raw:.4f}**")



# FOOTER

st.divider()

st.caption(
"""
GET324 Laboratory Mini Project

Model: Custom CNN

Developed for educational purposes only.
"""
)
