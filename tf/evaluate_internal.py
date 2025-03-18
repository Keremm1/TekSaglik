import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import accuracy_score

# Kaydedilmiş modeli yükle
model_path = "stroke_detection_model.h5"
model = load_model(model_path)

# Dataset klasörü
data_dir = "dataset"
img_size = (128, 128)

# Klasörler ve etiketler
class_labels = {"Inme Var": 0, "Inme Yok": 1}

# Veri ve etiketleri saklamak için listeler
images = []
labels = []

# Klasörlerde gezerek resimleri yükle ve modele uygun hale getir
for class_name, label in class_labels.items():
    class_path = os.path.join(data_dir, class_name) #dataset/Inme Var
    for img_name in os.listdir(class_path): #1000.png
        img_path = os.path.join(class_path, img_name) #dataset/Inme Var/1000.png
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img) / 255.0  # Normalizasyon
        images.append(img_array)
        labels.append(label)

# Listeyi numpy array'e çevir
images = np.array(images)
labels = np.array(labels)

# Model tahminleri
predictions = model.predict(images)
predictions = (predictions > 0.5).astype(int).flatten()  # 0 veya 1'e dönüştür

# Accuracy hesapla
accuracy = accuracy_score(labels, predictions)
print(f"Model Accuracy: {accuracy:.4f}")
