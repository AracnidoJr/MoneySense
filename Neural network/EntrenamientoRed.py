# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 09:49:20 2025

@author: ivanv
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Aumentación de datos con transformaciones adicionales
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.2, 1.0],  # Ajuste de brillo para mayor variabilidad
    fill_mode='nearest',
    validation_split=0.2)

validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Cargar el modelo preentrenado MobileNetV2 sin las capas superiores
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Descongelar algunas capas del modelo preentrenado para un ajuste más fino
base_model.trainable = True
fine_tune_at = 60  # Descongelamos las últimas capas
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Construir el modelo con la parte preentrenada + nuevas capas
model = models.Sequential([
    base_model,  # Capa base preentrenada
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(8, activation='softmax')  # 7 clases (ajusta si hay más clases)
])

# Compilar el modelo con una tasa de aprendizaje más baja
initial_learning_rate = 1e-4  # Tasa de aprendizaje más baja
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Directorio donde están tus imágenes
dataset_directory = 'C:/Users/ivanv/OneDrive/Documents/ITAM/Procesamiento Digital de Señales/Proyecto/Dataset Billetes'

# Generadores de imágenes para entrenamiento y validación
train_generator = train_datagen.flow_from_directory(
    dataset_directory,  # Asegúrate de que esta ruta es correcta
    target_size=(224, 224),
    batch_size=64,
    class_mode='sparse',  # Usa 'sparse' si las etiquetas son números enteros
    subset='training')

validation_generator = validation_datagen.flow_from_directory(
    dataset_directory,
    target_size=(224, 224),
    batch_size=64,
    class_mode='sparse',
    subset='validation')

# Entrenamiento del modelo (más épocas)
history = model.fit(
    train_generator,
    epochs=35,  # Aumentar las épocas
    validation_data=validation_generator)

# Guardar el modelo entrenado
model.save('modelo_billetes_mobilenetv2_finetuned.keras')
