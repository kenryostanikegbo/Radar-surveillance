"""
Calibration pass v2: accuracy + margin diagnostics on the held-out Test split.

Goals:
  - Run the deployed INT8 model (the actual file on ESP32-S3) on
    real RadHAR Test humans + synthetic vehicles/unknown.
  - Report not just accuracy but also LOGIT MARGINS -- the model's
    raw scores before softmax. These are the true confidence signal.
  - softmax([1, 0, 0]) always gives ~0.58 / 0.21 / 0.21 = 0.36 margin,
    which is structural noise. We work in logit space.

Usage:
    python calibration_v2.py
"""

import os
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


CLASS_NAMES = ["human", "vehicle", "unknown"]


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def eval_split(interpreter, X, y, label, return_probs=False):
    """Run model over X, return (preds, logit_margins, top1_logits, lats)."""
    in_det  = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    in_s, in_z   = in_det.get("quantization", (1.0, 0))
    out_s, out_z = out_det.get("quantization", (1.0, 0))

    preds, logit_margins, top1_logits, lats = [], [], [], []
    for i in range(len(X)):
        x_q = (X[i:i+1] / in_s + in_z).clip(-128, 127).astype(np.int8)
        interpreter.set_tensor(in_det["index"], x_q)
        t0 = time.perf_counter()
        interpreter.invoke()
        lats.append((time.perf_counter() - t0) * 1000)
        # Logits (dequantized raw scores, no softmax)
        logits = (interpreter.get_tensor(out_det["index"]).astype(np.float32) - out_z) * out_s
        logits = logits[0]
        order = np.argsort(logits)[::-1]
        preds.append(int(order[0]))
        logit_margins.append(float(logits[order[0]] - logits[order[1]]))
        top1_logits.append(float(logits[order[0]]))

    preds = np.array(preds)
    logit_margins = np.array(logit_margins)
    top1_logits = np.array(top1_logits)
    acc = (preds == y).mean()
    print(f"  {label:<28} n={len(X):>5}  acc={acc*100:6.2f}%  "
          f"top1 logit mean={top1_logits.mean():.3f}  "
          f"margin mean={logit_margins.mean():.3f}  "
          f"margin p5={np.percentile(logit_margins, 5):.3f}  "
          f"lat p50={np.percentile(lats, 50):.3f}ms")

    if return_probs:
        return preds, logit_margins, top1_logits
    return preds, logit_margins, top1_logits


def per_activity_breakdown(interpreter, win_npz, norm, win_path):
    """Re-derive features from windows so we can break down by activity."""
    from compute_features import window_to_features

    win = np.load(win_path)
    X_w = win["tensors"]
    M_w = win["masks"]
    activities = win["activities"]

    feats = np.stack([window_to_features(t, m) for t, m in zip(X_w, M_w)])
    feats_n = (feats - norm["mean"]) / norm["std"]

    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    in_s, in_z = in_det.get("quantization", (1.0, 0))
    out_s, out_z = out_det.get("quantization", (1.0, 0))

    preds, logit_margins = [], []
    for x in feats_n:
        x_q = (x / in_s + in_z).clip(-128, 127).astype(np.int8)[None]
        interpreter.set_tensor(in_det["index"], x_q)
        interpreter.invoke()
        logits = (interpreter.get_tensor(out_det["index"]).astype(np.float32) - out_z) * out_s
        logits = logits[0]
        order = np.argsort(logits)[::-1]
        preds.append(int(order[0]))
        logit_margins.append(float(logits[order[0]] - logits[order[1]]))

    preds = np.array(preds)
    logit_margins = np.array(logit_margins)

    print(f"\n  Per-activity accuracy on real Test humans (n={len(preds)}):")
    print(f"  (logit margin: max ~1.0; near 1.0 = saturation-confident)")
    for act in np.unique(activities):
        m = activities == act
        if m.sum() == 0:
            continue
        sub_acc = (preds[m] == 0).mean()
        print(f"    {str(act):>8}: n={m.sum():>5}  acc={sub_acc*100:6.2f}%  "
              f"logit margin mean={logit_margins[m].mean():.3f}")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    norm = np.load(os.path.join(root, "models", "norm_params.npz"))
    tflite_path = os.path.join(root, "models", "model_int8.tflite")

    print(f"Model: {tflite_path}  ({os.path.getsize(tflite_path)/1024:.1f} KB)")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    print(f"Input  : {in_det['dtype']} shape={in_det['shape']}")

    # ---- Load and normalize each split ----
    real_h = np.load(os.path.join(root, "data", "real_humans_features_test.npz"))
    veh    = np.load(os.path.join(root, "data", "synthetic_vehicles_features.npz"))
    unk    = np.load(os.path.join(root, "data", "synthetic_unknown_features.npz"))

    X_real_h = (real_h["X"] - norm["mean"]) / norm["std"]
    X_v      = (veh["X"]    - norm["mean"]) / norm["std"]
    X_u      = (unk["X"]    - norm["mean"]) / norm["std"]

    y_real_h = np.zeros(len(X_real_h), dtype=int)
    y_v      = np.ones(len(X_v), dtype=int)
    y_u      = np.full(len(X_u), 2, dtype=int)

    # Subsample synthetics for speed
    rng = np.random.default_rng(123)
    def sub(X, y, n):
        n = min(n, len(X))
        idx = rng.choice(len(X), size=n, replace=False)
        return X[idx], y[idx]

    X_v_s, y_v_s   = sub(X_v, y_v, 1500)
    X_u_s, y_u_s   = sub(X_u, y_u, 1500)
    X_h_s, y_h_s   = sub(X_real_h, y_real_h, min(2000, len(X_real_h)))

    print("\nPer-split evaluation (accuracy + top1 + margin):")
    p_h, m_h, pr_h = eval_split(interpreter, X_h_s, y_h_s, "real RadHAR Test humans")
    p_v, m_v, pr_v = eval_split(interpreter, X_v_s, y_v_s, "synthetic vehicles")
    p_u, m_u, pr_u = eval_split(interpreter, X_u_s, y_u_s, "synthetic unknown")

    X_all = np.concatenate([X_h_s, X_v_s, X_u_s])
    y_all = np.concatenate([y_h_s, y_v_s, y_u_s])
    p_all = np.concatenate([p_h, p_v, p_u])
    m_all = np.concatenate([m_h, m_v, m_u])

    print(f"\n--- Combined held-out (n={len(X_all)}) ---")
    print(f"  Overall accuracy:        {(p_all == y_all).mean()*100:.2f}%")
    print(f"  Logit margin p5:         {np.percentile(m_all, 5):.3f}")
    print(f"  Logit margin p50:        {np.percentile(m_all, 50):.3f}")
    print(f"  Logit margin p95:        {np.percentile(m_all, 95):.3f}")
    print(f"  Low-margin (<0.2):       {(m_all < 0.2).sum()} ({(m_all < 0.2).mean()*100:.1f}%)")
    print(f"  Negative margin (wrong): {(m_all < 0).sum()}")

    print("\nClassification report (combined):")
    print(classification_report(y_all, p_all, target_names=CLASS_NAMES,
                                digits=3, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_all, p_all)
    print(f"        {'  '.join(f'{c:>8}' for c in CLASS_NAMES)}")
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:>8} {'  '.join(f'{v:>8}' for v in row)}")

    # Per-activity breakdown on full Test set (recompute features from windows)
    print("\nFull Test set, per-activity breakdown:")
    per_activity_breakdown(
        interpreter, None, norm,
        os.path.join(root, "data", "rd03d_dataset_test.npz")
    )


if __name__ == "__main__":
    main()
