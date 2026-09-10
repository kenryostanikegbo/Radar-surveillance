"""
Layer 1 offline verification of the trained INT8 TFLite model.

Runs the actual quantized model that will go on ESP32-S3, on the PC,
to verify:
  - Accuracy on held-out test set
  - Per-class precision/recall/F1
  - Inference latency on PC (sanity check, not representative of S3)
  - Model file size fits ESP32-S3 budget (< 50 KB)

Usage:
    python test_model_offline.py \
        --model models/model_int8.tflite \
        --data data/combined_features.npz \
        --norm models/norm_params.npz
"""

import argparse
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


CLASS_NAMES = ["human", "vehicle", "unknown"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data",  required=True)
    p.add_argument("--norm",  required=True)
    p.add_argument("--n-test", type=int, default=0,
                   help="0 = use all test data; else random sample N rows")
    args = p.parse_args()

    # ---- File size check ----
    size_bytes = os.path.getsize(args.model)
    size_kb = size_bytes / 1024
    print(f"Model size: {size_bytes} bytes ({size_kb:.1f} KB)")
    if size_kb > 50:
        print(f"  [!]  WARNING: Model exceeds 50 KB ESP32-S3 budget!")
    else:
        print(f"  [OK]  Within ESP32-S3 budget")

    # ---- Load data and normalize ----
    data = np.load(args.data)
    X, y_str = data["X"], data["y"]
    norm = np.load(args.norm)
    X_n = (X - norm["mean"]) / norm["std"]

    label_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    y = np.array([label_to_idx[s] for s in y_str])

    # Take test split (same seed as training script for reproducibility)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X_n, y, test_size=0.2,
                                            stratify=y, random_state=42)

    if args.n_test > 0:
        idx = np.random.choice(len(X_test), size=args.n_test, replace=False)
        X_test = X_test[idx]
        y_test = y_test[idx]

    print(f"Test set: {X_test.shape[0]} samples")

    # ---- Load TFLite model ----
    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_dtype  = input_details["dtype"]
    output_dtype = output_details["dtype"]
    input_scale,  input_zero_point  = input_details.get("quantization", (1.0, 0))
    output_scale, output_zero_point = output_details.get("quantization", (1.0, 0))

    print(f"\nInput  details: {input_dtype}, shape {input_details['shape']}")
    print(f"Output details: {output_dtype}, shape {output_details['shape']}")

    # ---- Run inference ----
    latencies_ms = []
    preds = []
    for i in range(len(X_test)):
        x = X_test[i:i+1]
        t0 = time.perf_counter()
        if input_dtype == np.int8:
            x_q = (x / input_scale + input_zero_point).clip(-128, 127).astype(np.int8)
            interpreter.set_tensor(input_details["index"], x_q)
        else:
            interpreter.set_tensor(input_details["index"], x.astype(np.float32))
        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        if output_dtype == np.int8:
            out_f = (out.astype(np.float32) - output_zero_point) * output_scale
        else:
            out_f = out
        preds.append(int(np.argmax(out_f)))

    preds = np.array(preds)

    # ---- Report ----
    acc = (preds == y_test).mean()
    print(f"\nAccuracy:  {acc*100:.2f}%")
    print(f"PC latency: {np.mean(latencies_ms):.2f} ms (mean), "
          f"{np.percentile(latencies_ms, 95):.2f} ms (p95)")

    print("\nClassification report:")
    print(classification_report(y_test, preds, target_names=CLASS_NAMES, digits=3))

    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_test, preds)
    print(f"        {'  '.join(f'{c:>8}' for c in CLASS_NAMES)}")
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:>8} {'  '.join(f'{v:>8}' for v in row)}")

    # ---- Pass criteria ----
    print("\n--- Pass criteria check ---")
    print(f"  [{'[OK]' if size_kb <= 50 else '[X]'}] Model size <= 50 KB  ({size_kb:.1f} KB)")
    print(f"  [{'[OK]' if acc >= 0.80 else '[X]'}] Overall accuracy >= 80%  ({acc*100:.1f}%)")


if __name__ == "__main__":
    main()
