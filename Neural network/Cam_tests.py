# Pruebas.py (Corregido)
# -*- coding: utf-8 -*-
"""
Script de pruebas para modelo de clasificación de billetes,
con visualización corregida de imágenes PIL.
"""

import os
import cv2
import time
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import pygame
# —————————————— EDITA ESTAS RUTAS ——————————————
MODEL_PATH  = 'modelo_billetes_optimized.keras'
TEST_IMAGE  = r'../Imagenes de Prueba/100tard.png'
TEST_DIR    = r'../Imagenes de Prueba'
AUDIO_DIR  = r'../AudiosBilletes'

# ——————————————————————————————————————————————
target_size=(224, 224)
_ultimo_audio = ""
_tiempo_ultimo_audio = 0

# Mapeo de clases (debe coincidir con tu entrenamiento)
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


pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)

def play_audio(label, intervalo=4):
    global _ultimo_audio, _tiempo_ultimo_audio

    tiempo_actual = time.time()
    if label != _ultimo_audio or (tiempo_actual - _tiempo_ultimo_audio) > intervalo:
        try:
            pygame.mixer.music.load(f"../AudiosBilletes/{label}.mp3")
            pygame.mixer.music.play()
            _ultimo_audio = label
            _tiempo_ultimo_audio = tiempo_actual
        except Exception as e:
            print(f"[ERROR] No se pudo reproducir {label}.mp3: {e}")

def load_and_preprocess(img_path, target_size=(224, 224)):
    """Carga y preprocesa una imagen para el modelo."""
    img = image.load_img(img_path, target_size=target_size)
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0), img

def predict_single(model, img_path, top_k=3, show_img=True):
    """
    Realiza predicción en una sola imagen.
    Devuelve lista de (etiqueta, probabilidad).
    """
    x, img = load_and_preprocess(img_path, model.input_shape[1:3])
    preds = model.predict(x)[0]
    idxs = preds.argsort()[-top_k:][::-1]
    top_preds = [(CLASS_LABELS[i], float(preds[i])) for i in idxs]
    if show_img:
        # Convertir PIL Image a array antes de mostrar
        img_arr = np.asarray(img).astype('uint8')
        plt.imshow(img_arr)
        plt.axis('off')
        title = " | ".join([f"{label}: {prob*100:.1f}%" for label, prob in top_preds])
        plt.title(title)
        plt.show()
    return top_preds

def batch_predict(model, dir_path):
    """
    Recorre un directorio y predice todas las imágenes.
    Imprime en consola la etiqueta y probabilidad.
    """
    for fname in sorted(os.listdir(dir_path)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(dir_path, fname)
            label, prob = predict_single(model, img_path, top_k=1, show_img=False)[0]
            print(f"{fname}: {label} ({prob*100:.1f}%)")

def main_ish():
    # Carga el modelo
    print(f"Cargando modelo de: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    # Predicción individual
    if os.path.isfile(TEST_IMAGE):
        print("\n=== Predicción individual ===")
        preds = predict_single(model, TEST_IMAGE, top_k=3, show_img=True)
        for label, prob in preds:
            print(f"{label}: {prob*100:.1f}%")
    else:
        print(f"Imagen de prueba no encontrada: {TEST_IMAGE}")

    # Predicción por lotes
    if os.path.isdir(TEST_DIR):
        print("\n=== Predicción por lotes ===")
        batch_predict(model, TEST_DIR)
    else:
        print(f"Directorio de prueba no encontrado: {TEST_DIR}")

def main():
    # Carga el modelo
    print(f"Cargando modelo de: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    capture = cv.VideoCapture(0, cv.CAP_DSHOW)

    while True:
        isTrue, frame = capture.read()
        if not isTrue:
            break
        #cv2.imshow("Camara", frame)
        img = cv2.resize(frame, target_size)
        img = img.astype('float32') / 255.0  # normalizar si fue entrenado así
        img = np.expand_dims(img, axis=0)

        # Predicción individual
        pred = model.predict(img)
        index = np.argmax(pred)
        prob = pred[0][index]
        clase = CLASS_LABELS[index]

        # Mostrar en pantalla
        texto = f'Billete: ${clase} ({prob * 100:.1f}%)'
        cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostrar frame
        cv2.imshow('Detección de Billetes', frame)
        play_audio(clase)
        # Mostrar el frame con la predicción

        if cv.waitKey(20) & 0xFF == ord('d'):
            break
    capture.release()
    cv.destroyAllWindows()
    pygame.mixer.quit()
if __name__ == "__main__":
    main()
