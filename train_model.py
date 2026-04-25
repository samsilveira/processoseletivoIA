import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import numpy as np

def train():
    # Configura seeds para reprodutibilidade
    np.random.seed(42)
    tf.random.set_seed(42)

    # Carregamento e pré-processamento do dataset
    print("Carregando dataset MNIST...")
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    x_train = x_train.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
    x_test  = x_test.reshape((-1, 28, 28, 1)).astype("float32") / 255.0

    # Definição da arquitetura
    model = models.Sequential([
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),

        layers.Dense(10, activation="softmax"),
    ])

    # Compilação e treinamento
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    print("Iniciando treinamento...")
    model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=5,
        validation_split=0.1,
    )

    # Avaliação e salvamento do modelo
    score = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nAcurácia final no conjunto de teste: {score[1]:.4f}")

    model.save("model.h5")
    print("Modelo salvo com sucesso: model.h5")

if __name__ == "__main__":
    train()
