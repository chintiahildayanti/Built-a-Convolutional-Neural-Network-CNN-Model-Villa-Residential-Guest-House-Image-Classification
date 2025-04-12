from PIL import Image
import os
import numpy as np

dataset_dir = r"C:\Users\IqbalKaldera\OneDrive\Documents\Dibimbing\bukit_vista_images"
file_list = os.listdir(dataset_dir)
print(file_list)  # Menampilkan daftar file dalam folder

import os

dataset_dir = r"C:\Users\IqbalKaldera\OneDrive\Documents\Dibimbing\bukit_vista_images"
subfolder = 'villa'  # Pilih salah satu subfolder

subfolder_path = os.path.join(dataset_dir, subfolder)
file_list = os.listdir(subfolder_path)
print(file_list)  # Menampilkan daftar file dalam subfolder

from PIL import Image
from IPython.display import display

# Pilih salah satu gambar dari daftar
image_path = os.path.join(subfolder_path, file_list[260])  # Ambil file

# Buka dan tampilkan gambar
image = Image.open(image_path)
image

image_array = np.array(image)
image_array/255

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Preprocessing dan augmentasi data
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 20% data untuk validasi
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load data training
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),  # Resize gambar ke 150x150
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

print(train_generator.class_indices)

# Load data validasi
validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 kelas: villa, guest_house, residential
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

# Training model
history = model.fit(
    train_generator,
    steps_per_epoch= len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    epochs=20
)

history

# Simpan model
model.save("model_cnn.h5")