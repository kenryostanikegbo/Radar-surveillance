"""
Alternative model: 1D-CNN on the time-axis tensor directly.

Whereas the MLP consumes hand-crafted 72-dim features, this model
consumes the raw (16, 3, 4) window and learns its own features.

Trade-offs:
  - More accurate (typically +2-5% F1)
  - Larger model (~30-60 KB INT8 vs ~10-15 KB for MLP)
  - Slightly slower inference on ESP32-S3 (~15-30 ms vs ~5-10 ms)

Use this if you want maximum accuracy and the model still fits the
50 KB budget. Otherwise stick with the MLP.

Usage:
    python train_1dcnn.py \
        --data data/combined_windows.npz \
        --out models/
"""

import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


CLASS_NAMES = ["human", "vehicle", "unknown"]


def build_1dcnn(window_size=16, max_targets=3, n_features=4, n_classes=3):
    """
    Lightweight 1D-CNN over time.

    Input shape: (window_size, max_targets * n_features) = (16, 12)
      - We flatten the (3 targets × 4 features) into 12 channels per timestep.
      - 1D convolutions slide along the time axis.

    Conv1D ops are supported by ESP-NN accelerator on ESP32-S3.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window_size, max_targets * n_features)),
        tf.keras.layers.Conv1D(16, kernel_size=3, activation="relu", padding="same"),
        tf.keras.layers.Conv1D(16, kernel_size=3, activation="relu", padding="same"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to combined windows npz")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", default="models/")
    args = p.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    # Load windowed dataset (not features)
    data = np.load(args.data)
    tensors = data["tensors"]   # (N, 16, 3, 4)
    masks   = data["masks"]     # (N, 16, 3)
    classes = data["classes"]

    label_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    y = np.array([label_to_idx[c] for c in classes])

    # Flatten the (3 targets, 4 features) per timestep into 12 channels
    # Mask padded values (broadcast mask across the 4 feature channels per target)
    X = tensors.reshape(len(tensors), 16, 3 * 4)               # (N, 16, 12)
    mask_broadcast = np.repeat(masks, 4, axis=-1)              # (N, 16, 12)
    X = X * mask_broadcast                                    # zero out padded entries

    print(f"Window dataset: {tensors.shape}, flattened to {X.shape}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Build
    model = build_1dcnn()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.15,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        ],
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc*100:.1f}%")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=3))

    # Save
    keras_path = os.path.join(args.out, "model_1dcnn.keras")
    model.save(keras_path)
    print(f"Saved 1D-CNN model -> {keras_path}")
    print("Run quantize_export.py on this model next.")


if __name__ == "__main__":
    main()
