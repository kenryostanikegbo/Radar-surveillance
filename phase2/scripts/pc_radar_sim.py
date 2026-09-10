"""
RD-03D UART simulator.

Reads window-format data (the same .npz produced by
`convert_iclegend_to_rd03d.py` or any compatible (16, 3, 4) tensor
dataset), decodes each window back into per-frame target structs, and
streams them over a real (or virtual) serial port at the firmware's
expected frame rate.

The firmware (see firmware/main/radar_uart.c) listens for our exact
wire format and runs the same protocol encoder/decoder as this script.

Two output modes:
    --port COM7           Real serial port (ESP32-S3 via USB-UART)
    --loopback            In-process loopback via a `loop://` URL.
                          Used by tests/loopback_test.py for hardware-free
                          verification of the firmware pipeline.

Optional logging:
    --log out.csv         One row per emitted frame (timestamp, n_targets,
                          target[0..2] d/v/x/y).

Usage:
    python scripts/pc_radar_sim.py \\
        --in  data/rd03d_dataset.npz \\
        --port COM7 --baud 256000 --fps 20

    python scripts/pc_radar_sim.py \\
        --in  data/rd03d_dataset.npz \\
        --loopback --fps 20 --frames 5000
"""

import argparse
import csv
import os
import sys
import time
from typing import Optional

import numpy as np

# Make `scripts.protocol_encode` importable regardless of cwd.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from protocol_encode import (
    Frame,
    Target,
    encode_frame,
    frame_size,
)


def load_windows(npz_path: str):
    """Return (tensors, classes) where tensors is (N, 16, 3, 4).
    Supports either the raw rd03d_dataset.npz shape OR a per-window
    combined_windows.npz shape -- the latter has a 'classes' field
    that we don't need here, just tensors and (optionally) activities.
    """
    d = np.load(npz_path)
    if "tensors" not in d.files:
        raise ValueError(f"{npz_path} has no 'tensors' field; "
                         "expected (N, 16, 3, 4) data")
    tensors = d["tensors"]
    if tensors.ndim != 4 or tensors.shape[1:] != (16, 3, 4):
        raise ValueError(f"{npz_path}: tensor shape {tensors.shape}, "
                         "expected (N, 16, 3, 4)")
    classes = d["classes"] if "classes" in d.files else None
    return tensors, classes


def window_to_targets_per_frame(window):
    """Decompose a (16, 3, 4) window into a list of 16 Frame objects.

    Each frame's target list is built from the window's 3 target slots:
        target[k].range_cm = window[t, k, 0] * 100    (m -> cm)
        target[k].speed    = window[t, k, 1] * 100    (m/s -> cm/s)
        target[k].x_cm     = window[t, k, 2] * 100
        target[k].y_cm     = window[t, k, 3] * 100

    Targets with all-zero features (masked slot) are skipped; the
    remaining are emitted in sorted order (largest range first is the
    convention RD-03D uses on the wire; not strictly required by the
    firmware which simply iterates n_targets).
    """
    frames = []
    for t in range(16):
        ts_ms = t * 50  # 20 Hz -> 50 ms between frames
        targets = []
        for k in range(3):
            d, v, x, y = window[t, k]
            if d == 0.0 and v == 0.0 and x == 0.0 and y == 0.0:
                # mask == 0: padded slot
                continue
            r_cm = int(round(d * 100.0))
            v_cs = int(round(v * 100.0))
            x_cm = int(round(x * 100.0))
            y_cm = int(round(y * 100.0))
            angle_cdeg = 0
            if r_cm > 0:
                import math
                angle_cdeg = int(round(math.degrees(math.atan2(y_cm, x_cm)) * 100))
            targets.append(Target(
                range_cm   = max(0, min(800, r_cm)),
                speed_cm_s = max(-1500, min(1500, v_cs)),
                angle_cdeg = max(-9000, min(9000, angle_cdeg)),
                rcs        = 50,
                flags      = 0x03 if abs(v_cs) > 5 else 0x01,
                x_cm       = max(-800, min(800, x_cm)),
                y_cm       = max(-800, min(800, y_cm)),
                z_cm       = 0,
            ))
        # Sort by range desc so the closest target is last (faster to scan visually)
        # Firmware doesn't require this; convention.
        targets.sort(key=lambda x: -x.range_cm)
        frames.append(Frame(timestamp_ms=ts_ms, targets=targets))
    return frames


class _LoopbackSerial:
    """A python-only "serial port" pair used in --loopback mode.

    .write() appends bytes to a buffer; .read() drains bytes.
    .read_until_frame() extracts whole frames of expected size.

    This is enough to drive the firmware pipeline in tests/loopback_test.py
    without needing a real COM port or pyserial.
    """

    def __init__(self, label="loopback"):
        self._buf = bytearray()
        self.label = label
        self._writes = 0
        self._reads = 0

    def write(self, data: bytes) -> int:
        self._buf.extend(data)
        self._writes += len(data)
        return len(data)

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = len(self._buf)
        n = min(n, len(self._buf))
        out = bytes(self._buf[:n])
        del self._buf[:n]
        self._reads += len(out)
        return out

    def read_all(self) -> bytes:
        return self.read(len(self._buf))

    def read_until_magic(self, magic: bytes, frame_len: int) -> Optional[bytes]:
        """Find MAGIC, then return the next frame_len bytes if available."""
        while True:
            idx = self._buf.find(magic)
            if idx < 0:
                return None
            # Drop bytes before the magic (resync).
            if idx > 0:
                del self._buf[:idx]
            if len(self._buf) < frame_len:
                return None
            frame = bytes(self._buf[:frame_len])
            del self._buf[:frame_len]
            return frame

    def stats(self):
        return {"writes": self._writes, "reads": self._reads, "buffered": len(self._buf)}


def open_sink(port: Optional[str], loopback: bool, baud: int):
    """Return an object with .write(bytes) -> int."""
    if loopback:
        return _LoopbackSerial()
    if not port:
        raise ValueError("either --port or --loopback must be specified")
    try:
        import serial  # pyserial
    except ImportError as e:
        raise ImportError(
            "pyserial is required for real serial output. "
            "Install with: pip install pyserial"
        ) from e
    return serial.Serial(port, baudrate=baud, bytesize=8,
                         parity="N", stopbits=1, timeout=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in", dest="npz", required=True,
                    help="Path to rd03d_dataset.npz (tensors (N, 16, 3, 4))")
    ap.add_argument("--port", default=None,
                    help="Serial port (e.g. COM7 or /dev/ttyUSB0)")
    ap.add_argument("--baud", type=int, default=256000)
    ap.add_argument("--fps",  type=int, default=20,
                    help="Frame rate to stream at")
    ap.add_argument("--loopback", action="store_true",
                    help="Use in-process loopback instead of a real port "
                         "(used by tests/loopback_test.py).")
    ap.add_argument("--frames", type=int, default=0,
                    help="Total frames to emit (0 = loop the dataset indefinitely)")
    ap.add_argument("--log", default=None,
                    help="Optional CSV path to log per-frame stats")
    ap.add_argument("--max-windows", type=int, default=0,
                    help="Limit number of windows processed (0 = all)")
    args = ap.parse_args()

    tensors, _ = load_windows(args.npz)
    if args.max_windows > 0:
        tensors = tensors[:args.max_windows]
    n_windows = len(tensors)
    print(f"Loaded {n_windows} windows from {args.npz} ({tensors.shape})")

    sink = open_sink(args.port, args.loopback, args.baud)
    if not args.loopback:
        print(f"Streaming at {args.fps} Hz on {args.port} @ {args.baud} baud")
    else:
        print(f"Loopback mode @ {args.fps} Hz, frame size = {frame_size(3)} bytes (max)")

    log_fp = open(args.log, "w", newline="") if args.log else None
    csv_w = None
    if log_fp:
        csv_w = csv.writer(log_fp)
        csv_w.writerow(["window_idx", "frame_idx", "timestamp_ms", "n_targets",
                        "t0_range_cm", "t0_speed_cm_s", "t0_x_cm", "t0_y_cm"])

    period_s = 1.0 / args.fps
    total_frames = 0
    next_deadline = time.perf_counter()
    window_iter = 0

    try:
        while True:
            if args.frames > 0 and total_frames >= args.frames:
                break

            # Within a window, emit each of the 16 frames at the requested FPS.
            window = tensors[window_iter % n_windows]
            frames = window_to_targets_per_frame(window)

            for fr in frames:
                if args.frames > 0 and total_frames >= args.frames:
                    break
                raw = encode_frame(fr)
                sink.write(raw)

                if csv_w is not None:
                    t0 = fr.targets[0] if fr.targets else None
                    csv_w.writerow([
                        window_iter, total_frames, fr.timestamp_ms, len(fr.targets),
                        t0.range_cm if t0 else 0,
                        t0.speed_cm_s if t0 else 0,
                        t0.x_cm if t0 else 0,
                        t0.y_cm if t0 else 0,
                    ])
                total_frames += 1

                # Pace at requested FPS.  Use a sleeping deadline loop so we
                # don't drift due to scheduler latency.
                next_deadline += period_s
                now = time.perf_counter()
                sleep_for = next_deadline - now
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    # Fell behind by >0 sec -- resync rather than fire-bursting.
                    next_deadline = now

            window_iter += 1
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if log_fp:
            log_fp.close()
        try:
            if hasattr(sink, "close"):
                sink.close()
        except Exception:
            pass

    if isinstance(sink, _LoopbackSerial):
        print(f"Loopback stats: {sink.stats()}")
    print(f"Emitted {total_frames} frames from {window_iter} windows")


if __name__ == "__main__":
    main()
