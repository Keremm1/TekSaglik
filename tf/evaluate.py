import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Modeli yükleyelim
model_path = 'stroke_detection_model.keras'
model = tf.keras.models.load_model(model_path)

# Test veri yolu
test_dir = 'test_dataset'

# ImageDataGenerator kullanarak test verisini hazırlayalım
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)  # Görselleri normalize ediyoruz

# Test verilerini başlatıyoruz
test_data_head_ct = []
test_labels_head_ct = []

test_data_brain_ct = []
test_labels_brain_ct = []

# head_ct ve brain_ct altındaki dizinleri gezelim
for folder in ['head_ct', 'brain_ct']:
    if folder == 'head_ct':
        test_data = test_data_head_ct
        test_labels = test_labels_head_ct
    else:
        test_data = test_data_brain_ct
        test_labels = test_labels_brain_ct

    folder_path = os.path.join(test_dir, folder)
    
    for label in ['Inme Var', 'Inme Yok']:
        label_folder = os.path.join(folder_path, label)
        
        # İlgili etiketler: 'Inme Var' -> 0, 'Inme Yok' -> 1
        label_value = 0 if label == 'Inme Var' else 1
        
        for img_file in os.listdir(label_folder):
            img_path = os.path.join(label_folder, img_file)
            
            # Görseli yükleyelim
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256))  # Modelin giriş boyutu (224,224) olabilir
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0  # Normalize et
            test_data.append(img_array)
            test_labels.append(label_value)

# Numpy dizilerine dönüştür
test_data_head_ct = np.array(test_data_head_ct)
test_labels_head_ct = np.array(test_labels_head_ct)

test_data_brain_ct = np.array(test_data_brain_ct)
test_labels_brain_ct = np.array(test_labels_brain_ct)

# head_ct için tahmin yapalım
predictions_head_ct = model.predict(test_data_head_ct)
predicted_labels_head_ct = (predictions_head_ct > 0.5).astype(int)

# brain_ct için tahmin yapalım
predictions_brain_ct = model.predict(test_data_brain_ct)
predicted_labels_brain_ct = (predictions_brain_ct > 0.5).astype(int)

# head_ct doğruluğunu hesaplayalım
accuracy_head_ct = accuracy_score(test_labels_head_ct, predicted_labels_head_ct)

# brain_ct doğruluğunu hesaplayalım
accuracy_brain_ct = accuracy_score(test_labels_brain_ct, predicted_labels_brain_ct)

# Toplam doğruluğu hesaplayalım
total_test_data = np.concatenate((test_data_head_ct, test_data_brain_ct), axis=0)
total_test_labels = np.concatenate((test_labels_head_ct, test_labels_brain_ct), axis=0)
total_predictions = np.concatenate((predicted_labels_head_ct, predicted_labels_brain_ct), axis=0)

accuracy_total = accuracy_score(total_test_labels, total_predictions)

# Sonuçları yazdıralım
print(f'Head CT Test Seti Doğruluğu: {accuracy_head_ct * 100:.2f}%')
print(f'Brain CT Test Seti Doğruluğu: {accuracy_brain_ct * 100:.2f}%')
print(f'Toplam Test Seti Doğruluğu: {accuracy_total * 100:.2f}%')
