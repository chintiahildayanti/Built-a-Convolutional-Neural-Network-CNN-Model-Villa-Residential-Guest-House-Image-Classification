import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = load_model("model_cnn.keras")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Mapping label (ganti sesuai urutan indeks dari train_generator.class_indices)
label_dict = {0: "guest house", 1: "residential", 2: "villa"}

# Judul aplikasi
st.title("Property Image Classification")

# Upload gambar
uploaded_file = st.file_uploader("Upload property image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded image', use_container_width=True)

    # Preprocessing
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    # Prediksi
    prediction = model.predict(x)
    class_index = np.argmax(prediction)
    class_label = label_dict[class_index]

 # Tampilkan hasil
    st.markdown("### Prediction:")
    st.success(f"This image is classified as: **{class_label}**")