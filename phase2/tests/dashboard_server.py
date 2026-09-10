"""
Local-PC dashboard. Runs the firmware dashboard HTML/JS/CSS at
http://127.0.0.1:8080 with live /api/state JSON powered by the same
loopback pipeline as tests/loopback_test.py.

What you get:
    - The exact browser view the ESP32-S3 will serve over its softAP.
    - Open http://127.0.0.1:8080 in any browser (the script opens it
      for you on Windows).
    - Tracks update in real time as the loopback pipeline runs.
    - All C firmware logic (parser, feature extractor) is exercised
      via the radar_parse.dll.

What it doesn't test:
    - The TFLite Micro inference on hardware (we use the NumPy MLP).
    - The actual Wi-Fi softAP network stack.

Usage:
    python tests/dashboard_server.py [--frames N] [--data FILE] [--shuffle]

The loop runs until --frames is reached, then idles with the final
state visible. Hit Ctrl-C to exit.
"""

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE2 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PHASE2, "scripts"))
sys.path.insert(0, os.path.join(PHASE2, "tests"))

# Reuse the loopback harness's pipeline builders.
from loopback_test import (
    load_dll, setup_dll_args, load_inference_model,
    quantize_features, load_windows,
)
import numpy as np


# Local copy of the loopback harness's Tracker (it lives inside main()
# there). Python port of firmware/tracker.c's alpha-beta filter.
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
            best, best_d2 = None, 4.0
            for i, d in enumerate(dets):
                if used[i]: continue
                dx, dy = tk.x - d["x"], tk.y - d["y"]
                d2 = dx * dx + dy * dy
                if d2 < best_d2: best_d2 = d2; best = i
            if best is not None:
                used[best] = True
                d = dets[best]
                rx, ry = d["x"] - tk.x, d["y"] - tk.y
                tk.x  += 0.7 * rx
                tk.y  += 0.7 * ry
                tk.vx += (0.5 / max(dt, 1e-3)) * rx
                tk.vy += (0.5 / max(dt, 1e-3)) * ry
                tk.speed = (tk.vx**2 + tk.vy**2) ** 0.5
                tk.last_seen_s = self.t
                tk.class_id = d["class_id"]
                tk.class_conf = d["class_conf"]
                tk.hit_count = getattr(tk, "hit_count", 0) + 1
                # Publish after 3 consecutive hits (~ 0.15s at 20 Hz,
                # ~ 0.6s at 5 Hz). Suppresses one-frame flicker.
                if tk.hit_count >= 3:
                    tk.published = True

        for i, d in enumerate(dets):
            if used[i]: continue
            tk = _Track()
            tk.id = self.next_id; self.next_id += 1
            tk.x, tk.y = d["x"], d["y"]
            tk.vx = tk.vy = 0
            tk.speed = d["speed"]
            tk.class_id = d["class_id"]
            tk.class_conf = d["class_conf"]
            tk.age_s = 0
            tk.last_seen_s = self.t
            tk.hit_count = 0
            tk.published = False
            self.tracks.append(tk)


class _Track:
    __slots__ = ("id", "x", "y", "vx", "vy", "speed", "class_id",
                 "class_conf", "age_s", "last_seen_s", "hit_count",
                 "published")

DASHBOARD_DIR = os.path.join(PHASE2, "firmware", "main", "dashboard")


# ---- Shared state written by the loopback thread, read by the
# ---- HTTP handler. Single-element box with a lock.
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = {
            "frame_count": 0,
            "infer_ms": 0.0,
            "free_heap": 0,
            "uptime_s": 0.0,
            "fps": 0.0,
            "tracks": [],
        }

    def update(self, **kwargs):
        with self.lock:
            self.state.update(kwargs)

    def snapshot(self):
        with self.lock:
            return dict(self.state)


def make_handler(state: SharedState):
    """Factory so the HTTP handler closes over the SharedState."""

    class Handler(BaseHTTPRequestHandler):
        # Quieter logging
        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self._serve_file("index.html", "text/html")
            elif self.path == "/app.js":
                self._serve_file("app.js", "application/javascript")
            elif self.path == "/style.css":
                self._serve_file("style.css", "text/css")
            elif self.path == "/api/state":
                body = json.dumps(state.snapshot()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/health":
                s = state.snapshot()
                body = json.dumps({
                    "uptime_s": s["uptime_s"],
                    "fps": s["fps"],
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def _serve_file(self, name, mime):
            path = os.path.join(DASHBOARD_DIR, name)
            try:
                with open(path, "rb") as f:
                    body = f.read()
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def run_loopback(state: SharedState, args):
    """Background loopback pipeline. Updates `state` as it goes."""
    lib = load_dll()
    setup_dll_args(lib)
    predict, model_kind = load_inference_model()
    if predict is None:
        raise RuntimeError("no usable MLP model")

    tensors, classes = load_windows(args.data)
    n_windows = len(tensors)
    if args.shuffle:
        rng = np.random.default_rng(args.shuffle_seed)
        order = rng.permutation(n_windows)
        tensors = tensors[order]
        classes = classes[order]

    norm = np.load(os.path.join(PHASE2, "models", "norm_params.npz"))
    feature_mean = norm["mean"]
    feature_std  = norm["std"]
    class_names  = ["human", "vehicle", "unknown"]
    class_label_to_idx = {"human": 0, "vehicle": 1, "unknown": 2}

    tracker = Tracker()

    import ctypes
    import struct

    c_targets_buf = (ctypes.c_uint8 * (16 * 3 * 16))()
    c_mask_buf    = (ctypes.c_uint8 * (16 * 3))()
    c_features    = (ctypes.c_float * 72)()

    period_s = 1.0 / args.fps
    total_frames = 0
    window_iter = 0
    frames_in_window = 0
    t_start = time.perf_counter()
    inf_ms_recent = 0.0
    frames_window_recent = 0
    fps_window_start = t_start
    fps_recent = 0.0
    parsed_ok = 0
    features_ok = 0
    inferences = 0
    correct = 0

    from protocol_encode import Target, encode_frame

    print(f"[loopback] backend={model_kind}, dataset={args.data}, "
          f"frames={args.frames}, shuffle={args.shuffle}")
    print(f"[loopback] serving dashboard at http://127.0.0.1:{args.port}/")

    while True:
        if 0 < args.frames <= total_frames:
            break
        if window_iter >= n_windows:
            # Cycle back to the start of the dataset (or stop if frames is finite).
            if args.frames > 0 and total_frames >= args.frames:
                break
            window_iter = 0
        win = tensors[window_iter]
        cur_class_label = classes[window_iter]
        window_iter += 1

        for t in range(16):
            if 0 < args.frames <= total_frames:
                break
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
            from protocol_encode import Frame
            fr = Frame(timestamp_ms=t * 50, targets=frame_targets)
            raw = encode_frame(fr)

            out_targets = (ctypes.c_uint8 * (16 * 3))()
            out_ts = ctypes.c_uint32(0)
            n = lib.radar_parse_frame(raw, len(raw), out_targets, ctypes.byref(out_ts))
            if n < 0:
                continue
            parsed_ok += 1

            base = t * 3 * 16
            for k in range(n):
                src = bytes(out_targets[k * 16:(k + 1) * 16])
                for b in range(16):
                    c_targets_buf[base + k * 16 + b] = src[b]
                c_mask_buf[t * 3 + k] = 1
            for k in range(n, 3):
                c_mask_buf[t * 3 + k] = 0

            frames_in_window += 1
            total_frames += 1

        if frames_in_window >= 16:
            frames_in_window = 0
            rc = lib.feature_extract_window(
                ctypes.cast(c_targets_buf, ctypes.c_void_p),
                ctypes.cast(c_mask_buf, ctypes.c_void_p),
                c_features,
            )
            if rc != 72:
                continue
            features_ok += 1

            features_f32 = np.array(c_features[:], dtype=np.float32)
            z = (features_f32 - feature_mean) / feature_std
            t0 = time.perf_counter()
            class_id, probs = predict(z)
            t1 = time.perf_counter()
            inf_ms_recent = (inf_ms_recent * inferences + (t1 - t0) * 1000.0) / (inferences + 1)
            inferences += 1

            expected = class_label_to_idx.get(cur_class_label, -1)
            if expected >= 0 and class_id == expected:
                correct += 1

            dets = []
            raw_buf = bytes(c_targets_buf)
            for tt in range(13, 16):
                for k in range(3):
                    if not c_mask_buf[tt * 3 + k]: continue
                    base_t = (tt * 3 + k) * 16
                    speed, x_cm, y_cm = struct.unpack_from("<hhh", raw_buf, base_t + 2)
                    dets.append({
                        "x": x_cm / 100.0, "y": y_cm / 100.0,
                        "speed": abs(speed / 100.0),
                        "class_id": class_id, "class_conf": float(probs[class_id]),
                    })
            tracker.step(period_s * 16, dets)

            # ---- Update shared state ----
            now = time.perf_counter()
            uptime = now - t_start
            frames_window_recent += 16
            if now - fps_window_start >= 1.0:
                fps_recent = frames_window_recent / (now - fps_window_start)
                fps_window_start = now
                frames_window_recent = 0

            tracks = []
            for tk in tracker.tracks[-16:]:
                # Show every track (active or published) so the dashboard
                # gives visual feedback even when the dataset's jitter
                # prevents the alpha-beta filter from settling. In real
                # deployment the radar reports a coherent scene at 20 Hz
                # and tracks stabilize after 3 hits.
                tracks.append({
                    "id": tk.id,
                    "class": class_names[tk.class_id],
                    "confidence": float(tk.class_conf),
                    "x": float(tk.x), "y": float(tk.y),
                    "speed": float(tk.speed),
                    "age_s": float(tk.age_s),
                    "published": bool(getattr(tk, "published", False)),
                    "hits": int(getattr(tk, "hit_count", 0)),
                })
            state.update(
                frame_count=parsed_ok,
                infer_ms=inf_ms_recent,
                uptime_s=uptime,
                fps=fps_recent,
                tracks=tracks,
            )

            # Pace at FPS so the dashboard gets a steady stream
            sleep_s = period_s * 16 - (time.perf_counter() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)

    # Final: print a one-line summary
    acc = 100.0 * correct / max(inferences, 1)
    print(f"[loopback] DONE: {parsed_ok} frames parsed, {features_ok} windows, "
          f"{inferences} inferences, accuracy={acc:.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=0,
                   help="Total frames to process (0 = idle forever)")
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--data", default="data/combined_windows.npz")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--shuffle-seed", type=int, default=42)
    p.add_argument("--port", type=int, default=8080,
                   help="HTTP port for the dashboard (default 8080)")
    p.add_argument("--no-browser", action="store_true",
                   help="Don't auto-open a browser tab")
    args = p.parse_args()

    state = SharedState()

    # HTTP server in its own daemon thread
    handler_cls = make_handler(state)
    httpd = HTTPServer(("127.0.0.1", args.port), handler_cls)
    server_thread = threading.Thread(target=httpd.serve_forever,
                                     daemon=True, name="httpd")
    server_thread.start()
    print(f"[dashboard] serving on http://127.0.0.1:{args.port}/")
    print(f"[dashboard]   GET /            -> firmware/main/dashboard/index.html")
    print(f"[dashboard]   GET /app.js      -> firmware/main/dashboard/app.js")
    print(f"[dashboard]   GET /style.css   -> firmware/main/dashboard/style.css")
    print(f"[dashboard]   GET /api/state   -> JSON, polled every 200 ms")
    print(f"[dashboard]   GET /api/health  -> JSON")

    if not args.no_browser:
        try:
            webbrowser.open(f"http://127.0.0.1:{args.port}/")
        except Exception:
            pass

    try:
        run_loopback(state, args)
        print("[dashboard] loopback finished. Press Ctrl-C to exit "
              "(the dashboard stays live with the final state).")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[dashboard] shutting down.")
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    main()
