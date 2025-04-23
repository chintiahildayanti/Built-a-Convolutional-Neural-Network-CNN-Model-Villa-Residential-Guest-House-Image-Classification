import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Opsional: matikan oneDNN jika ingin hasil numerik lebih stabil (tidak wajib)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load model hanya sekali dan kompilasi (hindari retracing)
@st.cache_resource
def load_my_model():
    model = load_model("model_cnn.keras")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_my_model()

# Mapping label (ganti sesuai urutan indeks dari train_generator.class_indices)
label_dict = {0: "guest house", 1: "residential", 2: "villa"}

# Fungsi prediksi dengan tf.function untuk efisiensi
@tf.function(reduce_retracing=True)
def predict_image(x):
    return model(x, training=False)

# Judul aplikasi
st.title("Bukit Vista Property Classifier")

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

    # Pastikan tipe input adalah tensor float32 (untuk tf.function)
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)

    # Prediksi
    prediction = predict_image(x_tensor)
    class_index = tf.argmax(prediction, axis=1).numpy()[0]
    class_label = label_dict[class_index]

    # Tampilkan hasil
    st.markdown("### Prediction:")
    st.success(f"This image is classified as: *{class_label}*")
    confidence = tf.reduce_max(prediction).numpy()
    st.info(f"Confidence: {confidence:.2%}")
