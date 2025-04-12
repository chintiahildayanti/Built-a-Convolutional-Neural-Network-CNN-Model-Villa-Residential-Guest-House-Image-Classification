import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import tempfile

st.set_page_config(page_title="Property Image Classification")

st.title("🏘️ Property Image Classification")

# Upload model
uploaded_model = st.file_uploader("Upload your trained model (.h5 or .keras)", type=["h5", "keras"])

# Upload image
uploaded_image = st.file_uploader("Upload property image", type=["jpg", "jpeg", "png"])

# Load model dari upload
model = None
if uploaded_model is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_model.read())
        model_path = tmp_file.name

    try:
        model = load_model(model_path, compile=False)  # Kalau hanya untuk prediksi, compile=False aman
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        model = None

# Prediksi gambar
if model is not None and uploaded_image is not None:
    img = Image.open(uploaded_image).convert('RGB')
    st.image(img, caption='Uploaded Image', use_container_width=True)

    # Preprocessing sesuai model
    img = img.resize((150, 150))  # ubah sesuai input model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Prediksi
    prediction = model.predict(img_array)
    class_names = ['Villa', 'Guest House', 'Residential']  # sesuaikan dengan class milikmu
    predicted_class = class_names[np.argmax(prediction)]

    st.subheader("🔍 Prediction")
    st.markdown(f"**This image is classified as:** `{predicted_class}`")
