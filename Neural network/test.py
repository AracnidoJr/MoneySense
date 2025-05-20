from tensorflow.keras.models import load_model

# Carga el modelo
model = load_model('modelo_billetes_optimized.keras')

# Muestra el resumen del modelo
model.summary()

# Imprime el tamaño de entrada y salida
print("Forma de entrada:", model.input_shape)
print("Forma de salida:", model.output_shape)
