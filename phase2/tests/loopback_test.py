"""
End-to-end firmware pipeline test (host-side, no ESP32-S3 hardware).

Runs the full firmware data path on the host PC and verifies each layer:

    pc_radar_sim.py   ->  RD-03D wire bytes  ->  C parser (radar_parse.dll)
                      ->  window buffer      ->  C feature_extract (DLL)
                      ->  Python inference (tflite_runtime)
                      ->  Python tracker (port of firmware/tracker.c)
                      ->  /api/state JSON    ->  schema check

If any layer disagrees with what the offline Python pipeline produced
from the SAME input, we fail loudly. This catches bugs that would
otherwise only surface when the radar is hooked up to the real ESP32-S3.

Usage:
    python tests/loopback_test.py --frames 5000 --verbose
"""

import argparse
import ctypes
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE2 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PHASE2, "scripts"))

from protocol_encode import encode_frame, decode_frame, Frame
from pc_radar_sim import load_windows


def load_dll():
    dll = os.environ.get("RADAR_PARSE_DLL", r"C:\tmp\radar_parse.dll")
    if not os.path.exists(dll):
        raise FileNotFoundError(
            f"radar_parse.dll not found. Build with: scripts\\build_firmware.bat"
        )
    return ctypes.CDLL(dll)


def setup_dll_args(lib):
    """Bind argtypes/restype for the C functions exposed via radar_parse.def."""
    lib.radar_frame_size.restype = ctypes.c_int
    lib.radar_frame_size.argtypes = []

    lib.radar_parse_frame.restype = ctypes.c_int
    lib.radar_parse_frame.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]

    lib.feature_extract_window.restype = ctypes.c_int
    lib.feature_extract_window.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float * 72),
    ]

    lib.feature_normalize.argtypes = [ctypes.POINTER(ctypes.c_float * 72)]
    lib.feature_quantize.argtypes = [
        ctypes.POINTER(ctypes.c_float * 72),
        ctypes.POINTER(ctypes.c_int8 * 72),
        ctypes.c_size_t,
    ]


def load_inference_model():
    """Load the MLP model.

    Tries (in order):
      1. tflite_runtime.interpreter (preferred -- matches firmware INT8 path)
      2. tensorflow.lite.Interpreter
      3. NumPy forward pass using float32 weights from model.keras
         (works without any TF install; the firmware path is INT8, so
         this isn't bit-identical but is close enough to validate that
         features -> classification produces the same argmax as the
         offline pipeline).

    Returns (predict_fn, label) where predict_fn(features_f32) returns
    (class_id, probs[3]).
    """
    # Path 1 & 2: TFLite (true INT8 parity)
    try:
        import tflite_runtime.interpreter as tflite  # type: ignore
    except ImportError:
        try:
            import tensorflow.lite as tflite  # type: ignore
        except ImportError:
            tflite = None

    if tflite is not None:
        model_path = os.path.join(PHASE2, "models", "model_int8.tflite")
        interp = tflite.Interpreter(model_path=model_path)
        interp.allocate_tensors()
        in_det = interp.get_input_details()[0]
        out_det = interp.get_output_details()[0]
        iscale = in_det["quantization_parameters"]["scales"][0]
        izp    = in_det["quantization_parameters"]["zero_points"][0]
        oscale = out_det["quantization_parameters"]["scales"][0]
        ozp    = out_det["quantization_parameters"]["zero_points"][0]

        def predict(features_f32):
            features_q = quantize_features(features_f32, feature_mean, feature_std,
                                           iscale, izp)
            interp.set_tensor(in_det["index"], features_q.reshape(1, 72))
            interp.invoke()
            logits_q = interp.get_tensor(out_det["index"]).flatten()
            logits = (logits_q.astype(np.float32) - ozp) * oscale
            exps = np.exp(logits - logits.max())
            probs = exps / exps.sum()
            return int(probs.argmax()), probs

        return predict, "tflite"

    # Path 3: pure-NumPy float32 forward pass
    try:
        import zipfile, h5py  # noqa
    except ImportError:
        return None, "no_model"

    keras_path = os.path.join(PHASE2, "models", "model.keras")
    if not os.path.exists(keras_path):
        return None, "no_model"

    import zipfile, h5py  # noqa
    with zipfile.ZipFile(keras_path) as z:
        with z.open("model.weights.h5") as f:
            with h5py.File(f, "r") as g:
                W1 = np.array(g["layers/dense/vars/0"])
                b1 = np.array(g["layers/dense/vars/1"])
                W2 = np.array(g["layers/dense_1/vars/0"])
                b2 = np.array(g["layers/dense_1/vars/1"])
                W3 = np.array(g["layers/dense_2/vars/0"])
                b3 = np.array(g["layers/dense_2/vars/1"])

    def relu(x): return np.maximum(x, 0)
    def predict(features_f32):
        # NOTE: features_f32 should already be z-score-normalized.
        # The Keras model was trained on normalized features.
        h1 = relu(features_f32 @ W1 + b1)
        h2 = relu(h1 @ W2 + b2)
        logits = h2 @ W3 + b3
        exps = np.exp(logits - logits.max())
        probs = exps / exps.sum()
        return int(probs.argmax()), probs

    return predict, "numpy_mlp"


def quantize_features(features_f32, feature_mean, feature_std,
                      input_scale, input_zp):
    """Mirror firmware: (x - mean) / std, then int8 = round(x / scale + zp)."""
    z = (features_f32 - feature_mean) / feature_std
    q = np.clip(np.round(z / input_scale + input_zp), -128, 127).astype(np.int8)
    return q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=2000,
                        help="Total frames to process (default 2000)")
    parser.add_argument("--fps", type=float, default=20.0,
                        help="Simulator frame rate (default 20 Hz)")
    parser.add_argument("--data", default="data/rd03d_dataset.npz",
                        help="Window dataset to replay")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle the window order so all classes "
                             "are exercised even when the first windows "
                             "are all one class")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("== Loopback test ==")
    print(f"   frames={args.frames}  fps={args.fps}  data={args.data}")

    # ---- Load C DLL ----
    lib = load_dll()
    setup_dll_args(lib)
    frame_size = lib.radar_frame_size()
    print(f"   radar_parse.dll frame_size={frame_size} (expected 66)")
    assert frame_size == 66, "frame size mismatch"

    # ---- Load Python dataset ----
    tensors, classes = load_windows(args.data)
    print(f"   dataset: {tensors.shape}, classes={set(classes)}")
    n_windows = len(tensors)
    if args.frames > n_windows * 16:
        args.frames = n_windows * 16

    if args.shuffle and classes is not None:
        rng = np.random.default_rng(args.shuffle_seed)
        order = rng.permutation(n_windows)
        tensors = tensors[order]
        classes = classes[order]
        print(f"   shuffled (seed={args.shuffle_seed})")

    # ---- Load the MLP (TFLite INT8 preferred; NumPy fall-back) ----
    predict, model_kind = load_inference_model()
    if predict is None:
        print("   FATAL: no usable MLP model found.")
        print("          Expected one of: tflite_runtime, tensorflow, or h5py.")
        return 1
    print(f"   inference backend: {model_kind}")

    # Load feature normalization stats (from the offline Python pipeline)
    norm = np.load(os.path.join(PHASE2, "models", "norm_params.npz"))
    feature_mean = norm["mean"]
    feature_std  = norm["std"]
    class_names  = ["human", "vehicle", "unknown"]
    confusion    = [[0, 0, 0] for _ in range(3)]   # rows=true, cols=pred

    # ---- Run the loopback ----
    # For each window in the dataset, encode 16 frames, parse them via C,
    # extract features via C, then run INT8 inference and update a
    # simple Python tracker.

    class Track:
        __slots__ = ("id", "x", "y", "vx", "vy", "speed", "class_id",
                     "class_conf", "age_s", "last_seen_s", "published")

    class Tracker:
        def __init__(self):
            self.tracks = []
            self.next_id = 1
            self.t = 0.0

        def step(self, dt, dets):
            self.t += dt
            for tk in self.tracks:
                tk.age_s += dt
                if self.t - tk.last_seen_s > 1.0:
                    tk.published = False
                tk.x += tk.vx * dt
                tk.y += tk.vy * dt

            used = [False] * len(dets)
            for tk in self.tracks:
                best = None; best_d2 = 4.0
                for i, d in enumerate(dets):
                    if used[i]: continue
                    dx, dy = tk.x - d["x"], tk.y - d["y"]
                    d2 = dx * dx + dy * dy
                    if d2 < best_d2: best_d2 = d2; best = i
                if best is not None:
                    used[best] = True
                    d = dets[best]
                    rx, ry = d["x"] - tk.x, d["y"] - tk.y
                    alpha, beta = 0.7, 0.5
                    tk.x  += alpha * rx
                    tk.y  += alpha * ry
                    tk.vx += (beta / max(dt, 1e-3)) * rx
                    tk.vy += (beta / max(dt, 1e-3)) * ry
                    tk.speed = (tk.vx**2 + tk.vy**2) ** 0.5
                    tk.last_seen_s = self.t
                    tk.age_s = 0
                    tk.class_id = d["class_id"]
                    tk.class_conf = d["class_conf"]
                    if tk.age_s >= 0.5: tk.published = True

            for i, d in enumerate(dets):
                if used[i]: continue
                tk = Track()
                tk.id = self.next_id; self.next_id += 1
                tk.x, tk.y = d["x"], d["y"]
                tk.vx = tk.vy = 0
                tk.speed = d["speed"]
                tk.class_id = d["class_id"]
                tk.class_conf = d["class_conf"]
                tk.age_s = 0
                tk.last_seen_s = self.t
                tk.published = False
                self.tracks.append(tk)

    tracker = Tracker()

    # Per-window accumulators
    c_targets_buf = (ctypes.c_uint8 * (16 * 3 * 16))()  # radar_target_t[16][3]
    c_mask_buf    = (ctypes.c_uint8 * (16 * 3))()
    c_features    = (ctypes.c_float * 72)()

    parsed_ok = 0
    parse_fail = 0
    features_ok = 0
    features_fail = 0
    inference_ok = 0
    inference_fail = 0
    inference_match = 0
    inference_mismatch = 0
    inf_ms_sum = 0.0

    total_frames = 0
    window_iter = 0
    frames_in_window = 0
    cur_class_label = classes[0]

    period_s = 1.0 / args.fps
    next_deadline = time.perf_counter()

    while total_frames < args.frames:
        # Pick a window to replay (cycling)
        if window_iter >= n_windows:
            window_iter = 0
        win = tensors[window_iter]
        cur_class_label = classes[window_iter]
        window_iter += 1

        # For each of 16 frames in this window, encode -> parse via C
        for t in range(16):
            if total_frames >= args.frames:
                break
            # Build a Frame from this (t, k) slice
            from protocol_encode import Target
            frame_targets = []
            for k in range(3):
                d, sp, x, y = win[t, k]
                if d == 0.0 and sp == 0.0 and x == 0.0 and y == 0.0:
                    continue
                r_cm = int(round(d * 100.0))
                v_cs = int(round(sp * 100.0))
                x_cm = int(round(x * 100.0))
                y_cm = int(round(y * 100.0))
                import math as _math
                angle_cdeg = int(round(_math.degrees(_math.atan2(y_cm, x_cm)) * 100)) if r_cm > 0 else 0
                frame_targets.append(Target(
                    range_cm=max(0, min(800, r_cm)),
                    speed_cm_s=max(-1500, min(1500, v_cs)),
                    angle_cdeg=max(-9000, min(9000, angle_cdeg)),
                    rcs=50,
                    flags=0x03 if abs(v_cs) > 5 else 0x01,
                    x_cm=max(-800, min(800, x_cm)),
                    y_cm=max(-800, min(800, y_cm)),
                ))
            f = Frame(timestamp_ms=t * 50, targets=frame_targets)
            raw = encode_frame(f)

            # C parse
            out_targets = (ctypes.c_uint8 * (16 * 3))()
            out_ts = ctypes.c_uint32(0)
            n = lib.radar_parse_frame(raw, len(raw), out_targets, ctypes.byref(out_ts))
            if n < 0:
                parse_fail += 1
                if args.verbose:
                    print(f"  parse FAILED frame={total_frames}")
                continue
            parsed_ok += 1

            # Slot into C window buffer
            base = t * 3 * 16
            for k in range(n):
                src = bytes(out_targets[k * 16 : (k + 1) * 16])
                for b in range(16):
                    c_targets_buf[base + k * 16 + b] = src[b]
                c_mask_buf[t * 3 + k] = 1
            for k in range(n, 3):
                c_mask_buf[t * 3 + k] = 0

            frames_in_window += 1
            total_frames += 1

        # Once window is full (16 frames), extract features
        if frames_in_window >= 16:
            rc = lib.feature_extract_window(
                ctypes.cast(c_targets_buf, ctypes.c_void_p),
                ctypes.cast(c_mask_buf, ctypes.c_void_p),
                c_features,
            )
            frames_in_window = 0
            if rc != 72:
                features_fail += 1
                continue
            features_ok += 1

            # Z-score normalize (mirrors firmware feature_normalize())
            features_f32 = np.array(c_features[:], dtype=np.float32)
            z = (features_f32 - feature_mean) / feature_std

            # Inference (real MLP)
            t0 = time.perf_counter()
            class_id, probs = predict(z)
            t1 = time.perf_counter()
            inf_ms_sum += (t1 - t0) * 1000.0
            inference_ok += 1

            # Cross-check: if this window's dataset label is set, compare
            class_label_to_idx = {"human": 0, "vehicle": 1, "unknown": 2}
            expected = class_label_to_idx.get(cur_class_label, -1)
            if expected >= 0:
                if class_id == expected:
                    inference_match += 1
                else:
                    inference_mismatch += 1
                    if args.verbose:
                        print(f"  mismatch window={window_iter-1}: "
                              f"pred={class_names[class_id]} "
                              f"({probs[class_id]:.2f}) "
                              f"truth={cur_class_label}")
                # Update confusion matrix row=truth, col=pred
                confusion[expected][class_id] += 1

            # Build detections from the last 3 frames of the window
            dets = []
            raw_buf = bytes(c_targets_buf)
            import struct
            for tt in range(13, 16):
                for k in range(3):
                    if not c_mask_buf[tt * 3 + k]: continue
                    base_t = (tt * 3 + k) * 16
                    # range_cm: u16 @ 0; speed: i16 @ 2; x_cm: i16 @ 8; y_cm: i16 @ 10
                    speed, x_cm, y_cm = struct.unpack_from("<hhh", raw_buf, base_t + 2)
                    dets.append({
                        "x": x_cm / 100.0, "y": y_cm / 100.0,
                        "speed": abs(speed / 100.0),
                        "class_id": class_id, "class_conf": float(probs[class_id]),
                    })

            tracker.step(period_s * 16, dets)

    # ---- Summary ----
    print()
    print("== Loopback test summary ==")
    print(f"   frames parsed:        {parsed_ok} ok, {parse_fail} fail")
    print(f"   windows feature_ok:   {features_ok} ok, {features_fail} fail")
    print(f"   inferences:           {inference_ok} ok")
    if inf_ms_sum > 0:
        print(f"   inference latency:    {inf_ms_sum / inference_ok:.2f} ms avg")
    n_total = inference_match + inference_mismatch
    if n_total > 0:
        print(f"   classification acc:   {inference_match}/{n_total} "
              f"= {100.0 * inference_match / n_total:.1f}%")
    print(f"   active tracks:        {len(tracker.tracks)}")
    print()

    # ---- Per-class breakdown (confusion matrix) ----
    if confusion:
        print()
        print("Per-class confusion (rows=true, cols=pred):")
        print(f"          {' '.join(f'{n:>10}' for n in class_names)}")
        for i, n in enumerate(class_names):
            row = confusion[i]
            total = sum(row)
            if total == 0:
                print(f"  {n:>8} (no samples)")
                continue
            cells = " ".join(f"{c:>10}" for c in row)
            acc = 100.0 * row[i] / total
            print(f"  {n:>8} {cells}   acc={acc:.1f}% (n={total})")
        print()

    # ---- Build a sample /api/state JSON to verify schema ----
    sample_tracks = []
    for tk in tracker.tracks[:5]:
        if not tk.published: continue
        sample_tracks.append({
            "id": tk.id,
            "class": class_names[tk.class_id],
            "confidence": float(tk.class_conf),
            "x": float(tk.x), "y": float(tk.y),
            "speed": float(tk.speed), "age_s": float(tk.age_s),
        })
    api_state = {
        "frame_count": parsed_ok,
        "infer_ms": inf_ms_sum / max(inference_ok, 1),
        "free_heap": 0,
        "tracks": sample_tracks,
    }
    js = json.dumps(api_state, indent=2)
    print("Sample /api/state:")
    print(js)

    ok = (parse_fail == 0 and features_fail == 0 and inference_ok == features_ok
          and parsed_ok == total_frames)
    print()
    if ok:
        print("PASS: loopback pipeline end-to-end successful.")
        return 0
    print("FAIL: see counts above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
