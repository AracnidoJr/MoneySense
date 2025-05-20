# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 09:46:58 2025

@author: ivanv
"""

# Pruebas_solo_una_imagen.py
# -*- coding: utf-8 -*-
"""
Clasifica UNA imagen de billete y reproduce el audio correspondiente.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import pygame

# ———————————— EDITA ESTAS RUTAS ————————————
MODEL_PATH  = 'modelo_billetes_optimized.keras'
TEST_IMAGE  = r'../Imagenes de Prueba/100tard.png'
TEST_DIR    = r'../Imagenes de Prueba'
AUDIO_DIR  = r'../AudiosBilletes'
# ————————————————————————————————————————————

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

# ----- AUDIO -----
audio_files = {lbl: os.path.join(AUDIO_DIR, f'{lbl}.mp3')
               for lbl in CLASS_LABELS.values()}

pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)

def play_audio(label):
    path = audio_files.get(label)
    if path and os.path.isfile(path):
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
# -----------------

def load_and_preprocess(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0), img

def predict_and_speak(model, img_path, top_k=3, threshold=0.40, show_img=True):
    x, pil_img = load_and_preprocess(img_path, model.input_shape[1:3])
    preds = model.predict(x, verbose=0)[0]
    idxs  = preds.argsort()[-top_k:][::-1]
    top   = [(CLASS_LABELS[i], float(preds[i])) for i in idxs]

    if show_img:
        plt.imshow(np.asarray(pil_img).astype('uint8'))
        plt.axis('off')
        plt.title(" | ".join([f"{l}: {p*100:.1f}%" for l, p in top]))
        plt.show()

    best_label, best_prob = top[0]
    if best_prob >= threshold:
        play_audio(best_label)

    return top

def main():
    if not os.path.isfile(TEST_IMAGE):
        raise FileNotFoundError(f"Imagen no encontrada: {TEST_IMAGE}")

    print(f"Cargando modelo: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("\n=== Predicción ===")
    results = predict_and_speak(model, TEST_IMAGE, top_k=3, threshold=0.40, show_img=True)
    aud_label = ""
    for label, prob in results:
        print(f"{label}: {prob*100:.1f}%")
        if (prob * 100) > 60:
            aud_label = r'{label}.mp3'
    while True:
        play_audio("aud_label")

if __name__ == "__main__":
    main()
