# Import library yang dibutuhkan
from PIL import Image    # Untuk manipulasi gambar
import os        # Untuk interaksi dengan sistem file
import numpy as np    # Untuk operasi numerik
import matplotlib.pyplot as plt        # Untuk visualisasi grafik
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay    # Evaluasi model
import tensorflow as tf    # Untuk membangun dan melatih model deep learning
from tensorflow.keras.preprocessing.image import ImageDataGenerator    # Untuk augmentasi dan preprocessing gambar
from tensorflow.keras.models import Sequential    # Untuk membuat model CNN
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Dropout    # Layer-layer CNN (arsitektur model)
from tensorflow.keras.models import load_model    # Untuk memuat model CNN
import random    # Untuk mengatur seed random
from tensorflow.keras.optimizers import Adam    # Optimizer

# Path ke folder dataset
dataset_dir = r"C:\Users\IqbalKaldera\OneDrive\Documents\Dibimbing\bukit_vista_images"

# Fungsi untuk menetapkan seed agar eksperimen dapat direproduksi (hasil konsisten)
def set_seed(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)    # Set seed ke 42

seed_value = 42    # Seed value yang konsisten untuk ImageDataGenerator

# Preprocessing dan augmentasi data
datagen = ImageDataGenerator(
    rescale=1./255,        # Normalisasi piksel ke range [0, 1]
    validation_split=0.2,    # 20% data untuk validasi
    rotation_range=20,        # Rotasi acak gambar
    width_shift_range=0.2,    # Pergeseran lebar secara acak
    height_shift_range=0.2,    # Pergeseran tinggi secara acak
    shear_range=0.2,        # Transformasi shear
    zoom_range=0.2,    # Zoom gambar secara acak
    horizontal_flip=True,    # Balik horizontal
    fill_mode='nearest'    # Mode pengisian piksel kosong
)

# Generator untuk data training
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),    # Ukuran input gambar
    batch_size=16,        # Jumlah gambar per batch
    class_mode='categorical',    # Output berupa one-hot vector
    subset='training',    # Ambil bagian training
    seed=seed_value,    # Seed agar shuffle konsisten
    shuffle=True    # Acak data
)

validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),    # Ukuran input gambar
    batch_size=16,                # Jumlah gambar per batch
    class_mode='categorical',    # Output berupa one-hot vector
    subset='validation',        # Ambil bagian validasi
    seed=seed_value,        # Seed agar shuffle konsisten
    shuffle=False        # Tidak diacak (dibutuhkan untuk evaluasi)
)

# Definisi arsitektur model CNN
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),    # Layer konvolusi pertama dengan 16 filter berukuran 3x3, Aktivasi ReLU digunakan untuk menambahkan non-linearitas, input_shape adalah gambar RGB berukuran 224x224 piksel.
    AveragePooling2D(2, 2),    # Layer pooling dengan ukuran pool 2x2 , tujuannya untuk mengurangi dimensi spasial (downsampling) dan ekstraksi fitur dominan
    Conv2D(32, (3, 3), activation='relu'),    # Layer konvolusi kedua dengan 32 filter berukuran 3x3, Meningkatkan jumlah filter agar model dapat belajar fitur yang lebih kompleks
    AveragePooling2D(2, 2),        # Pooling untuk mengurangi dimensi dari output layer sebelumnya
    Conv2D(64, (3, 3), activation='relu'),    # Layer konvolusi ketiga dengan 64 filter
    AveragePooling2D(2, 2),    # Pooling lagi untuk mengecilkan dimensi dan mengurangi beban komputasi
    Conv2D(128, (3, 3), activation='relu'),    # Layer konvolusi keempat dengan 128 filter
    AveragePooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu'),    # Layer konvolusi kelima dengan 256 filter
    AveragePooling2D(2, 2),
    Flatten(),                        # Flatten layer untuk masuk ke dense
    Dense(128, activation='relu'),    # Fully connected layer
    Dropout(0.5),                     # Dropout untuk mencegah overfitting
    Dense(3, activation='softmax')    # Output layer dengan 3 kelas
])

# Kompilasi model dengan optimizer Adam dan loss categorical crossentropy
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())    # Tampilkan ringkasan arsitektur model

# Latih model menggunakan data training dan validasi
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),    # Langkah per epoch sesuai jumlah batch training
    validation_data=validation_generator,
    validation_steps=len(validation_generator),    # Langkah validasi sesuai jumlah batch validasi
    epochs=30    # Jumlah epoch pelatihan
)

# Simpan model hasil pelatihan ke file .keras
model.save("model_cnn.keras")

# ============================================
# === EVALUASI PERFORMA MODEL DI SINI ===
# ============================================

# Evaluasi akurasi dan loss pada data validasi
val_loss, val_acc = model.evaluate(validation_generator)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")

# Visualisasi akurasi dan loss selama pelatihan
plt.figure(figsize=(12, 5))

# Plot akurasi
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Prediksi untuk Confusion Matrix dan Classification Report
y_true = validation_generator.classes                # Label sebenarnya
y_pred_probs = model.predict(validation_generator)    # Probabilitas hasil prediksi
y_pred = np.argmax(y_pred_probs, axis=1)                # Ambil label dengan probabilitas tertinggi

# Confusion Matrix
class_labels = list(validation_generator.class_indices.keys())    # Label kelas dalam urutan indeks
cm = confusion_matrix(y_true, y_pred)        # Hitung confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap='Blues')        # Tampilkan confusion matrix dengan colormap biru
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))
