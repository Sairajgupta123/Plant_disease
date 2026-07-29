import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

model = load_model()

# -----------------------------
# Load Class Names
# -----------------------------
class_names = np.load("class_names.npy", allow_pickle=True)

IMG_SIZE = (224, 224)

# -----------------------------
# Prediction Function
# -----------------------------
def predict(image):

    image = image.resize(IMG_SIZE)

    img_array = tf.keras.preprocessing.image.img_to_array(image)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions)

    confidence = float(np.max(predictions)) * 100

    predicted_class = class_names[predicted_index]

    return predicted_class, confidence


# -----------------------------
# UI
# -----------------------------
st.title("🌿 Plant Disease Detection")

st.write(
    "Upload a plant leaf image and the model will predict the disease."
)

uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):

        with st.spinner("Predicting..."):

            disease, confidence = predict(image)

        st.success("Prediction Completed")

        st.subheader("Prediction")

        st.write(f"**Disease:** {disease}")

        st.write(f"**Confidence:** {confidence:.2f}%")

        st.progress(int(confidence))