import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data_dir = "dataset"  # Ana veri klasörü
img_size = (256, 256)  # Resimleri yeniden boyutlandırma
batch_size = 32

data_gen = ImageDataGenerator(rescale=1./255) #validation_split=0.2 # Normalizasyon

train_data = data_gen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

# val_data = data_gen.flow_from_directory(
#     data_dir,
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode='binary',
#     subset='validation'
# )

# # Örnek resim ve etiketleri gösterme
# batch_images, batch_labels = next(train_data)
# fig, axes = plt.subplots(2, 5, figsize=(15, 6))
# axes = axes.ravel()
# for i in range(10):
#     axes[i].imshow(batch_images[i])
#     axes[i].set_title(f"Etiket: {int(batch_labels[i])}")
#     axes[i].axis('off')
# plt.tight_layout()
# plt.show()

# Basit CNN Modeli
def create_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary Classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = create_model()

# Modeli Eğitme
history = model.fit(train_data,  epochs=10) #validation_data=val_data,

# Eğitim sonuçlarını görselleştirme
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
#plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()

# Modeli Kaydetme
model.save("stroke_detection_model.keras")
