# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 19:34:43 2025

@author: ivanv
Detección en tiempo real de billetes con Raspberry Pi + OpenCV + TensorFlow
y reproducción de audio según la predicción.

"""

import os
import time
import cv2
import numpy as np
try:
    # Si usas TFLite (más ligero en Pi), descomenta:
    # from tflite_runtime.interpreter import Interpreter
    # MODEL_PATH = 'modelo_billetes_optimized.tflite'
    # interpreter = Interpreter(model_path=MODEL_PATH)
    # interpreter.allocate_tensors()
    # input_details  = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    # def predict(img):
    #     interpreter.set_tensor(input_details[0]['index'], img)
    #     interpreter.invoke()
    #     return interpreter.get_tensor(output_details[0]['index'])[0]
    import tensorflow as tf
    MODEL_PATH = 'modelo_billetes_optimized.keras'
    model = tf.keras.models.load_model(MODEL_PATH)
    def predict(img):
        return model.predict(img)[0]
except Exception as e:
    raise RuntimeError(f"Error cargando modelo: {e}")

# Mapeo de clases y ficheros de audio
CLASS_LABELS = {
    0: '1000A_Pesos',
    1: '1000_Pesos',
    2: '100_Pesos',
    3: '200_Pesos',
    4: '20A_Pesos',
    5: '20_Pesos',
    6: '500A_Pesos',
    7: '500_Pesos',
    8: '50A_Pesos',
    9: '50_Pesos'
}
AUDIO_DIR = 'audio'
audio_files = {
    label: os.path.join(AUDIO_DIR, f"{label}.mp3")
    for label in CLASS_LABELS.values()
}

# Inicializa audio
import pygame
pygame.mixer.init()

def play_audio(label):
    """Carga y reproduce el mp3 correspondiente al label."""
    path = audio_files.get(label)
    if path and os.path.isfile(path):
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()

# Configuración de captura de cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara")

# Parámetros de preprocesamiento
TARGET_SIZE = (224, 224)

last_spoken = None
COOLDOWN = 2.0  # segundos mínimos entre audios idénticos
last_time = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocesamiento para el modelo
        img = cv2.resize(frame, TARGET_SIZE)
        img_norm = img.astype('float32') / 255.0
        inp = np.expand_dims(img_norm, axis=0)

        # Predicción
        preds = predict(inp)
        idx = np.argmax(preds)
        label = CLASS_LABELS[idx]
        prob  = preds[idx]

        # Si la confianza es razonable y es distinto o timeout, reproduce audio
        now = time.time()
        if prob > 0.6 and (label != last_spoken or now - last_time > COOLDOWN):
            play_audio(label)
            last_spoken = label
            last_time = now

        # Mostrar resultado en la ventana
        text = f"{label} {prob*100:4.1f}%"
        cv2.putText(frame, text, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Detector de Billetes", frame)

        # Salir con ESC
        if cv2.waitKey(1) == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
