import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_split(split_name: str):
    x_path = os.path.join("processed", split_name, "X.npy")
    y_path = os.path.join("processed", split_name, "y.npy")

    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.float32)

    return X, y


def build_model(n_mels=64, time_frames=101):
    inputs = keras.Input(shape=(n_mels, time_frames, 1))

    x = layers.BatchNormalization()(inputs)

    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def main():
    set_seed(42)

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    print("Train:", X_train.shape, y_train.shape)
    print("Val:  ", X_val.shape, y_val.shape)
    print("Test: ", X_test.shape, y_test.shape)

    model = build_model(
        n_mels=X_train.shape[1],
        time_frames=X_train.shape[2],
    )

    os.makedirs("checkpoints", exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath="checkpoints/best_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nEvaluating on test set...")
    results = model.evaluate(X_test, y_test, verbose=1)

    print("\nTest metrics:")
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.4f}")

    model.save("keyboard_click_model.keras")
    print("\nSaved final model to keyboard_click_model.keras")


if __name__ == "__main__":
    main()