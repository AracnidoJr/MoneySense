# -*- coding: utf-8 -*-
"""
Entrenamiento de clasificación de billetes con MobileNetV2
Ajustes: augmentación muy moderada + ruido, AdamW + Label Smoothing,
dos fases (cabeza → fine-tuning), EarlyStopping en val_loss.
Created on 2025-04-27
@author: ivanv
"""

import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import CategoricalCrossentropy

# Parámetros
IMAGE_SIZE  = (224, 224)
BATCH_SIZE  = 16
NUM_CLASSES = 10
DATA_DIR    = 'C:/Users/ivanv/OneDrive/Documents/ITAM/Procesamiento Digital de Senales/Proyecto/Dataset Billetes'
SEED        = 123

assert os.path.isdir(DATA_DIR), f"Ruta no existe: {DATA_DIR}"

# 1) Generadores con augmentación muy moderada + ruido gaussiano
def add_gaussian_noise(img):
    noise = np.random.normal(loc=0.0, scale=0.01, size=img.shape)
    return img + noise

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,             # ±5°
    width_shift_range=0.05,       # ±5%
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    brightness_range=(0.95, 1.05),
    preprocessing_function=add_gaussian_noise,  # ruido de cámara
    fill_mode='nearest',
    validation_split=0.2
)
validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',     # one-hot para label smoothing
    subset='training',
    seed=SEED
)
validation_generator = validation_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=SEED
)

# 2) Backbone preentrenado
base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3)
)
base_model.trainable = False  # Fase 1: solo cabeza

# 3) Construcción de la cabeza con Dropout y L2
inputs = layers.Input((*IMAGE_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)                                            # menos dropout
x = layers.Dense(512, activation='relu',
                 kernel_regularizer=regularizers.l2(1e-6))(x)         # menos L2
x = layers.Dropout(0.2)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = models.Model(inputs, outputs)

# 4) Pérdida y optimizador: AdamW + Label Smoothing
loss_fn = CategoricalCrossentropy(label_smoothing=0.05)

# 5) Callbacks fase 1
def lr_schedule1(epoch):
    max_lr = 1e-3
    if epoch < 5:
        return (epoch + 1) / 5 * max_lr
    progress = (epoch - 5) / (25 - 5)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))

lr_callback1 = LearningRateScheduler(lr_schedule1)
early_stop1 = EarlyStopping(
    monitor='val_loss', patience=7, mode='min', restore_best_weights=True
)
checkpoint1 = ModelCheckpoint(
    'best_phase1.keras', monitor='val_accuracy',
    save_best_only=True, mode='max'
)

# 6) Fase 1: entrenar solo la cabeza
model.compile(
    optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-5),
    loss=loss_fn,
    metrics=['accuracy']
)

history1 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=25,
    callbacks=[lr_callback1, early_stop1, checkpoint1]
)

# 7) Fase 2: fine-tuning de las últimas 20 capas del backbone
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=AdamW(learning_rate=5e-6, weight_decay=1e-5),
    loss=loss_fn,
    metrics=['accuracy']
)

early_stop2 = EarlyStopping(
    monitor='val_loss', patience=4, mode='min', restore_best_weights=True
)
checkpoint2 = ModelCheckpoint(
    'best_phase2.keras', monitor='val_accuracy',
    save_best_only=True, mode='max'
)

history2 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    callbacks=[early_stop2, checkpoint2]
)

# 8) Guardar modelo final
model.save('modelo_billetes_optimized.keras')
