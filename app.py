# Import Library yang dibutuhkan
import os        # Untuk manipulasi sistem dan environment
import streamlit as st        # Untuk membangun aplikasi web interaktif
import tensorflow as tf        # Untuk pemrosesan model deep learning
from tensorflow.keras.models import load_model    # Untuk memuat model yang telah dilatih
from tensorflow.keras.preprocessing import image    # Untuk preprocessing gambar
import numpy as np    # Untuk manipulasi array numerik
from PIL import Image    # Untuk membuka dan memproses file gambar

# Nonaktifkan oneDNN agar hasil komputasi lebih stabil (terutama saat debugging)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load model hanya sekali dan kompilasi (hindari retracing)
@st.cache_resource        # Cache resource agar model tidak dimuat berulang
def load_my_model():
    model = load_model("model_cnn.keras")    # Muat model dari file .keras
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])    # Kompilasi model
    return model    # Kembalikan model

model = load_my_model()    # Panggil fungsi untuk memuat model

# Dictionary untuk mapping indeks hasil prediksi ke nama kelas (ganti sesuai urutan indeks dari train_generator.class_indices)
label_dict = {0: "guest house", 1: "residential", 2: "villa"}

# Fungsi prediksi dengan dekorator tf.function agar lebih efisien dan cepat
@tf.function(reduce_retracing=True)    # Hindari retracing saat fungsi dipanggil berulang
def predict_image(x):
    return model(x, training=False)    # Lakukan prediksi tanpa training (inference mode)

# Judul aplikasi Streamlit
st.title("Bukit Vista Property Classifier")

# Komponen upload gambar dari pengguna
uploaded_file = st.file_uploader("Upload property image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:    # Jika ada file yang diupload
    img = Image.open(uploaded_file).convert('RGB')    # Buka gambar dan ubah ke format RGB
    st.image(img, caption='Uploaded image', use_container_width=True)    # Tampilkan gambar di Streamlit

    # Resize dan preprocessing gambar agar sesuai dengan input model
    img = img.resize((224, 224))    # Resize ke ukuran input model
    x = image.img_to_array(img)    # Konversi gambar ke array numpy
    x = np.expand_dims(x, axis=0)    # Tambahkan dimensi batch (1, 224, 224, 3)
    x = x / 255.0        # Normalisasi pixel ke range 0-1

    # Konversi array ke tensor TensorFlow bertipe float32
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)

    # Prediksi kelas gambar menggunakan model
    prediction = predict_image(x_tensor)    # Dapatkan prediksi dari model
    class_index = tf.argmax(prediction, axis=1).numpy()[0]    # Ambil indeks kelas dengan probabilitas tertinggi
    class_label = label_dict[class_index]    # Konversi indeks ke label kelas

    # Tampilkan hasil prediksi ke pengguna
    st.markdown("### Prediction:")
    st.success(f"This image is classified as: *{class_label}*")    # Tampilkan nama kelas hasil prediksi
    
    # Tampilkan confidence (tingkat keyakinan model)
    confidence = tf.reduce_max(prediction).numpy()    # Ambil nilai confidence tertinggi dari output softmax
    st.info(f"Confidence: {confidence:.2%}")    # Tampilkan confidence dalam persentase
