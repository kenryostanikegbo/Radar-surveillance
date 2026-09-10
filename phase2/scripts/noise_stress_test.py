"""
Noise-stress test: how does the deployed INT8 model degrade as the input
gets noisier?

Real RD-03D returns won't be as clean as RadHAR or our synthetic
generators -- they have range-bin quantization, multi-path, antenna
cross-talk, etc. This script progressively adds Gaussian noise to
features and reports when accuracy drops below thresholds.

This calibrates how robust our model is -- if it tolerates sigma=0.5
feature noise with 95% accuracy, but breaks at sigma=1.0, we know the
"headroom" before deploying to real hardware.

Usage:
    python noise_stress_test.py
"""

import os
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score


CLASS_NAMES = ["human", "vehicle", "unknown"]


def run_with_noise(interpreter, X_n, y, sigma):
    """Add Gaussian noise sigma to each feature, evaluate."""
    in_det  = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    in_s, in_z   = in_det.get("quantization", (1.0, 0))
    out_s, out_z = out_det.get("quantization", (1.0, 0))

    rng = np.random.default_rng(0)
    preds = []
    for i in range(len(X_n)):
        x = X_n[i].copy()
        if sigma > 0:
            x = x + rng.normal(0, sigma, size=x.shape).astype(np.float32)
        x_q = (x / in_s + in_z).clip(-128, 127).astype(np.int8)[None]
        interpreter.set_tensor(in_det["index"], x_q)
        interpreter.invoke()
        out = interpreter.get_tensor(out_det["index"])
        logits = (out.astype(np.float32) - out_z) * out_s
        preds.append(int(np.argmax(logits[0])))
    return accuracy_score(y, preds)


def per_class_noise(interpreter, X_n, y, sigma):
    in_det  = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    in_s, in_z   = in_det.get("quantization", (1.0, 0))
    out_s, out_z = out_det.get("quantization", (1.0, 0))
    rng = np.random.default_rng(0)
    by_class = {c: [0, 0] for c in CLASS_NAMES}  # [correct, total]
    for i in range(len(X_n)):
        x = X_n[i].copy()
        if sigma > 0:
            x = x + rng.normal(0, sigma, size=x.shape).astype(np.float32)
        x_q = (x / in_s + in_z).clip(-128, 127).astype(np.int8)[None]
        interpreter.set_tensor(in_det["index"], x_q)
        interpreter.invoke()
        out = interpreter.get_tensor(out_det["index"])
        logits = (out.astype(np.float32) - out_z) * out_s
        pred = int(np.argmax(logits[0]))
        cls = CLASS_NAMES[y[i]]
        by_class[cls][1] += 1
        if pred == y[i]:
            by_class[cls][0] += 1
    return {c: v[0]/v[1] if v[1] else 0 for c, v in by_class.items()}


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    norm = np.load(os.path.join(root, "models", "norm_params.npz"))
    tflite_path = os.path.join(root, "models", "model_int8.tflite")

    print(f"Model: {tflite_path}")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Build a balanced held-out set
    real = np.load(os.path.join(root, "data", "real_humans_features_test.npz"))
    veh  = np.load(os.path.join(root, "data", "synthetic_vehicles_features.npz"))
    unk  = np.load(os.path.join(root, "data", "synthetic_unknown_features.npz"))
    rng = np.random.default_rng(7)

    def take(X, y_label, n):
        n = min(n, len(X))
        idx = rng.choice(len(X), size=n, replace=False)
        return X[idx], np.full(n, CLASS_NAMES.index(y_label))

    X_h, y_h = take(real["X"], "human",   1000)
    X_v, y_v = take(veh["X"],  "vehicle", 1000)
    X_u, y_u = take(unk["X"],  "unknown", 1000)

    X = np.concatenate([X_h, X_v, X_u]).astype(np.float32)
    y = np.concatenate([y_h, y_v, y_u])
    X_n = (X - norm["mean"]) / norm["std"]

    # Normalized features have unit variance-ish, so sigma is in "stdev units".
    # sigma=0.1 ~ 10% of feature stdev; sigma=1.0 = 1 sigma of noise per feature.
    sigmas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

    print(f"\nBaseline (n={len(X)}, balanced 1000/class)")
    print("Noise sweeps on normalized features:")
    print(f"  {'sigma':>6}  {'acc':>7}  {'human':>7}  {'vehicle':>7}  {'unknown':>7}")
    for sigma in sigmas:
        acc = run_with_noise(interpreter, X_n, y, sigma)
        if sigma in (0.0, 0.5, 1.0, 1.5, 2.0):
            pc = per_class_noise(interpreter, X_n, y, sigma)
            print(f"  {sigma:>6.2f}  {acc*100:>6.2f}%  "
                  f"{pc['human']*100:>6.2f}%  {pc['vehicle']*100:>6.2f}%  "
                  f"{pc['unknown']*100:>6.2f}%")
        else:
            print(f"  {sigma:>6.2f}  {acc*100:>6.2f}%")


if __name__ == "__main__":
    main()
