# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 16:10:54 2025

@author: ivanv
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Cargar el modelo entrenado
model = tf.keras.models.load_model('modelo_billetes_mobilenetv2_finetuned.keras')

# Cargar una imagen nueva
img_path =  'C:/Users/ivanv/OneDrive/Documents/ITAM/Procesamiento Digital de Señales/Proyecto/Imagenes de Prueba/100tard.png' # Cambia esta ruta por la imagen que deseas probar
img = image.load_img(img_path, target_size=(224, 224))  # Redimensiona la imagen

# Convertir la imagen a un array numpy y escalar los valores
img_array = image.img_to_array(img) / 255.0  # Asegúrate de aplicar la misma escala que en el entrenamiento

# Añadir una dimensión extra para que sea compatible con la entrada del modelo (batch size 1)
img_array = np.expand_dims(img_array, axis=0)

# Realizar la predicción
predictions = model.predict(img_array)

# Obtener la clase predicha
predicted_class = np.argmax(predictions, axis=1)

# Mapeo de clases a sus nombres (asumimos que las clases están mapeadas como en 'class_indices')
class_labels = {0: '1000_Pesos', 1: '100_Pesos', 2: '200_Pesos', 3: '20_Pesos', 
                4: '500A_Pesos', 5: '500_Pesos', 6: '50_Pesos'}

# Imprimir la clase predicha
print(f"La imagen ha sido categorizada como: {class_labels[predicted_class[0]]}")
