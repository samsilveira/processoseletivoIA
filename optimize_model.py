import tensorflow as tf
import os

def optimize():
    # Carrega o modelo original
    model_path = "model.h5"
    if not os.path.exists(model_path):
        print("Erro: model.h5 não encontrado.")
        return

    model = tf.keras.models.load_model(model_path)

    # Configura conversão com Dynamic Range Quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Salva o modelo otimizado
    tflite_path = "model.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(tflite_path) / 1024
    print(f"Modelo TFLite gerado com sucesso!")
    print(f"Tamanho final do modelo: {size_kb:.2f} KB")

if __name__ == "__main__":
    optimize()
