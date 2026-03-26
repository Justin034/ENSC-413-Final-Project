from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def build_model(n_mels=64, time_frames=101):
    inputs = keras.Input(shape=(n_mels, time_frames, 1))

    x = layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    return model


def load_data():
    """
    Replace this dummy data with your real dataset later.
    Expected shapes:
      X_train: (N, 64, 101, 1)
      y_train: (N,)
      X_val:   (M, 64, 101, 1)
      y_val:   (M,)
    """
    X_train = np.random.rand(100, 64, 101, 1).astype(np.float32)
    y_train = np.random.randint(0, 2, size=(100,)).astype(np.float32)

    X_val = np.random.rand(20, 64, 101, 1).astype(np.float32)
    y_val = np.random.randint(0, 2, size=(20,)).astype(np.float32)

    return X_train, y_train, X_val, y_val


def main():
    X_train, y_train, X_val, y_val = load_data()

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)

    model = build_model(
        n_mels=X_train.shape[1],
        time_frames=X_train.shape[2]
    )

    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            "keyboard_detector.keras",
            monitor="val_loss",
            save_best_only=True
        )
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=16,
        callbacks=callbacks
    )

    model.save("keyboard_detector.keras")
    print("Training complete.")


if __name__ == "__main__":
    main()