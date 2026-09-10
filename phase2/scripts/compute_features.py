"""
Convert (window, mask) tensor dataset to flat feature vectors for training.

Each window (16 frames x 3 targets x 4 values) becomes a fixed-size
feature vector. This is what the ESP32-S3 model will consume.

Features (per window):
  - 3 targets x 17 per-target features = 51
  - 21 global scene features = 21
                            total: 72 features

Why these features?
  - Humans have higher speed variance (limb motion)
  - Vehicles have higher speed mean (driving) + lower variability per frame
  - Clutter has low persistence (intermittent detections)
  - Humans have characteristic Doppler variance (micro-Doppler signature)

Usage:
    python compute_features.py \
        --in data/rd03d_dataset.npz \
        --out data/features.npz
"""

import argparse
import numpy as np


FEATS_PER_TARGET = 17
GLOBAL_FEATS = 21
TOTAL_FEATS = 3 * FEATS_PER_TARGET + GLOBAL_FEATS  # = 72


def window_to_features(tensor, mask):
    """
    tensor : (16, 3, 4) -- d, speed, x, y per target per frame
    mask   : (16, 3)    -- 1 if real target, 0 if padded

    Returns fixed-size 72-dim feature vector.
    """
    feats = []

    # ---- per-target statistics over the window ----
    for k in range(3):
        present = mask[:, k]
        if present.sum() < 1:
            # target never present: zero features
            feats.extend([0.0] * FEATS_PER_TARGET)
            continue

        d = tensor[:, k, 0]
        v = tensor[:, k, 1]
        x = tensor[:, k, 2]
        y = tensor[:, k, 3]

        d_present = d[present > 0.5]
        v_present = v[present > 0.5]
        x_present = x[present > 0.5]
        y_present = y[present > 0.5]

        target_feats = []

        # 1-4. Mean values
        target_feats.append(float(d_present.mean()))
        target_feats.append(float(v_present.mean()))
        target_feats.append(float(x_present.mean()))
        target_feats.append(float(y_present.mean()))

        # 5-8. Standard deviation (variability -- humans have HIGH due to micro-Doppler)
        if len(d_present) > 1:
            target_feats.append(float(d_present.std()))
            target_feats.append(float(v_present.std()))
            target_feats.append(float(x_present.std()))
            target_feats.append(float(y_present.std()))
        else:
            target_feats.extend([0.0, 0.0, 0.0, 0.0])

        # 9-10. Slope (linear trend) of speed and range
        if len(d_present) > 1:
            t_idx = np.arange(len(d_present))
            target_feats.append(float(np.polyfit(t_idx, d_present, 1)[0]))
            target_feats.append(float(np.polyfit(t_idx, v_present, 1)[0]))
        else:
            target_feats.extend([0.0, 0.0])

        # 11. Max absolute speed (humans rarely > 2 m/s; vehicles up to 15)
        target_feats.append(float(np.abs(v_present).max()))

        # 12. Persistence ratio -- fraction of frames the target was seen
        target_feats.append(float(present.sum() / 16.0))

        # 13. Speed sign changes (humans oscillate, vehicles consistent)
        if len(v_present) > 1:
            sign_changes = float(np.sum(np.diff(np.sign(v_present)) != 0))
            target_feats.append(sign_changes / max(1, len(v_present) - 1))
        else:
            target_feats.append(0.0)

        # 14. Trajectory smoothness -- mean squared acceleration
        if len(v_present) > 2:
            accel = np.diff(v_present)
            target_feats.append(float(np.mean(accel ** 2)))
        else:
            target_feats.append(0.0)

        # 15-16. Spatial extent (how far the target moved in x,y)
        if len(x_present) > 1:
            target_feats.append(float(x_present.max() - x_present.min()))
            target_feats.append(float(y_present.max() - y_present.min()))
        else:
            target_feats.extend([0.0, 0.0])

        # 17. Average distance to (0,0)
        target_feats.append(float(np.sqrt(x_present.mean()**2 + y_present.mean()**2)))

        assert len(target_feats) == FEATS_PER_TARGET, \
            f"Expected {FEATS_PER_TARGET} features per target, got {len(target_feats)}"
        feats.extend(target_feats)

    # ---- global scene statistics (21 features) ----
    total_detections = float(mask.sum())
    feats.append(total_detections / (16 * 3))            # overall detection density
    feats.append(float(mask[:, 0].sum()) / 16)           # primary target persistence
    feats.append(float(mask[:, 1].sum()) / 16)           # secondary target persistence
    feats.append(float(mask[:, 2].sum()) / 16)           # tertiary target persistence

    # Mean / std range across all targets in this window
    d_all = tensor[:, :, 0][mask > 0.5]
    feats.append(float(d_all.mean()) if len(d_all) > 0 else 0.0)
    feats.append(float(d_all.std())  if len(d_all) > 1 else 0.0)

    # Mean / std / max speed across all targets
    v_all = tensor[:, :, 1][mask > 0.5]
    feats.append(float(v_all.mean()) if len(v_all) > 0 else 0.0)
    feats.append(float(v_all.std())  if len(v_all) > 1 else 0.0)
    feats.append(float(np.abs(v_all).max()) if len(v_all) > 0 else 0.0)

    # Frames where at least one target was present
    frames_with_target = float((mask.sum(axis=1) > 0).sum())
    feats.append(frames_with_target / 16)

    # Average number of targets per frame
    feats.append(total_detections / 16 / 3)

    # Speed spread (interquartile range)
    if len(v_all) > 1:
        feats.append(float(np.percentile(v_all, 75) - np.percentile(v_all, 25)))
    else:
        feats.append(0.0)

    # Range change rate
    if len(d_all) > 1:
        feats.append(float(np.mean(np.abs(np.diff(d_all)))))
    else:
        feats.append(0.0)

    # Speed change rate
    if len(v_all) > 1:
        feats.append(float(np.mean(np.abs(np.diff(v_all)))))
    else:
        feats.append(0.0)

    # Fractions at different speed bands (humans ~0-2, vehicles 5-15)
    feats.append(float((np.abs(v_all) > 0.5).mean()) if len(v_all) > 0 else 0.0)
    feats.append(float((np.abs(v_all) > 2.0).mean()) if len(v_all) > 0 else 0.0)
    feats.append(float((np.abs(v_all) > 5.0).mean()) if len(v_all) > 0 else 0.0)
    feats.append(float((np.abs(v_all) < 0.1).mean()) if len(v_all) > 0 else 0.0)

    # Max simultaneous targets in a single frame
    feats.append(float(mask.sum(axis=1).max()) / 3)

    # Fraction of empty frames (no targets detected)
    feats.append(float((mask.sum(axis=1) == 0).sum()) / 16)

    # Average speed magnitude across all detections
    feats.append(float(np.abs(v_all).mean()) if len(v_all) > 0 else 0.0)

    assert len(feats) == TOTAL_FEATS, \
        f"Expected {TOTAL_FEATS} total features, got {len(feats)}"

    return np.array(feats, dtype=np.float32)


def build_features(npz_in, npz_out):
    """Process an entire windowed dataset, save feature matrix."""
    data = np.load(npz_in)
    tensors = data["tensors"]
    masks = data["masks"]
    classes = data["classes"]

    X = np.stack([window_to_features(t, m) for t, m in zip(tensors, masks)])
    y = classes

    np.savez_compressed(npz_out, X=X, y=y)
    print(f"Features: {X.shape} from {len(tensors)} windows -> {npz_out}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Feature dim: {X.shape[1]}")
    return X, y


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    build_features(args.inp, args.out)
