"""
Train a lightweight MLP classifier on RD-03D-format features.

Designed for ESP32-S3 deployment:
  - Dense layers only (no convolutions, no recurrent)
  - Small model size (target: <50 KB INT8)
  - Fast inference (<10 ms on ESP32-S3)
  - 72-dim input -> 3-class softmax output

Usage:
    python train_mlp.py \
        --data data/combined_features.npz \
        --epochs 40 \
        --out models/
"""

import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


CLASS_NAMES = ["human", "vehicle", "unknown"]


def build_model(input_dim=72, n_classes=3):
    """
    Compact MLP. Final size after INT8 quantization: ~5-15 KB.

    Architecture choice:
      - 32 -> 16 -> 3 dense layers
      - Dropout 0.2 to prevent overfitting on small dataset
      - ReLU activations (well-supported by ESP-NN accelerator)
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,), name="input"),
        tf.keras.layers.Dense(32, activation="relu", name="dense_1"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu", name="dense_2"),
        tf.keras.layers.Dense(n_classes, activation="softmax", name="output"),
    ])
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to combined features npz")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", default="models/")
    args = p.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    data = np.load(args.data)
    X, y = data["X"], data["y"]

    # Encode labels to integers
    label_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    y_idx = np.array([label_to_idx[label] for label in y])

    print(f"Dataset: {X.shape}, classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_idx, test_size=0.2, stratify=y_idx, random_state=42
    )

    # --- Compute normalization stats from training set ---
    # Save these so the ESP32-S3 can apply identical scaling at inference
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-6
    X_train_n = (X_train - mean) / std
    X_test_n  = (X_test  - mean) / std

    # Save normalization params
    np.savez(os.path.join(args.out, "norm_params.npz"), mean=mean, std=std)
    print(f"Saved normalization params to {args.out}/norm_params.npz")

    # --- Build model ---
    model = build_model(input_dim=X_train.shape[1], n_classes=len(CLASS_NAMES))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # --- Train ---
    history = model.fit(
        X_train_n, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.15,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=8, restore_best_weights=True, verbose=1
            ),
        ],
    )

    # --- Evaluate ---
    test_loss, test_acc = model.evaluate(X_test_n, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc*100:.1f}%")

    y_pred = np.argmax(model.predict(X_test_n, verbose=0), axis=1)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_test, y_pred)
    print(f"        {'  '.join(f'{c:>8}' for c in CLASS_NAMES)}")
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:>8} {'  '.join(f'{v:>8}' for v in row)}")

    # --- Save Keras model ---
    keras_path = os.path.join(args.out, "model.keras")
    model.save(keras_path)
    print(f"\nSaved Keras model -> {keras_path}")


if __name__ == "__main__":
    main()
