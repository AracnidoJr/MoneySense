# Pruebas.py (Corregido)
# -*- coding: utf-8 -*-
"""
Script de pruebas para modelo de clasificación de billetes,
con visualización corregida de imágenes PIL.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# —————————————— EDITA ESTAS RUTAS ——————————————
MODEL_PATH  = 'modelo_billetes_optimized.keras'
TEST_IMAGE  = r'../Imagenes de Prueba/20_3.png'
TEST_DIR    = r'../Imagenes de Prueba'
# ——————————————————————————————————————————————

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

def main():
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

if __name__ == "__main__":
    main()
