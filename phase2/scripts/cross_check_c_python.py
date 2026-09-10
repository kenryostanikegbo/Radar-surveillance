"""
Cross-language agreement test for the radar wire protocol.

This is the source of truth that the firmware's CRC and parser behave
identically to the Python `protocol_encode.py` reference implementation.

It runs entirely on the PC: Python encodes frames, then both the Python
decoder AND the C decoder (radar_parse.dll) parse them. Every byte,
every field, every CRC must match.

Run:
    python scripts/cross_check_c_python.py
"""

import ctypes
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from protocol_encode import (
    encode_frame,
    decode_frame,
    Frame,
    Target,
    MAGIC,
    HEADER_SIZE,
    TARGET_SIZE,
    MAX_TARGETS,
    frame_size,
    crc16_ccitt,
)


def load_dll():
    dll_path = os.environ.get("RADAR_PARSE_DLL", r"C:\tmp\radar_parse.dll")
    if not os.path.exists(dll_path):
        raise FileNotFoundError(
            f"radar_parse.dll not found at {dll_path}. "
            f"Build it first with: scripts\\build_firmware.bat"
        )
    lib = ctypes.CDLL(dll_path)
    lib.radar_crc16.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    lib.radar_crc16.restype  = ctypes.c_uint16

    class CTarget(ctypes.Structure):
        _fields_ = [
            ("range_cm",   ctypes.c_uint16),
            ("speed_cm_s", ctypes.c_int16),
            ("angle_cdeg", ctypes.c_int16),
            ("rcs",        ctypes.c_uint8),
            ("flags",      ctypes.c_uint8),
            ("x_cm",       ctypes.c_int16),
            ("y_cm",       ctypes.c_int16),
            ("z_cm",       ctypes.c_int16),
            ("reserved",   ctypes.c_uint16),
        ]
    lib.radar_parse_frame.argtypes = [
        ctypes.c_char_p, ctypes.c_size_t,
        ctypes.POINTER(CTarget), ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.radar_parse_frame.restype = ctypes.c_int
    lib.radar_frame_size.argtypes = []
    lib.radar_frame_size.restype  = ctypes.c_int
    return lib, CTarget


def main():
    print("Loading radar_parse.dll...")
    lib, CTarget = load_dll()
    fsize_c = lib.radar_frame_size()
    fsize_py = frame_size(3)
    assert fsize_c == fsize_py, f"frame size mismatch: C={fsize_c} Py={fsize_py}"
    print(f"  Frame size: C={fsize_c}, Py={fsize_py}, MATCH")

    # 1. Reference CRC vector
    ref_c = lib.radar_crc16(b"123456789", 9)
    ref_py = crc16_ccitt(b"123456789")
    print(f"  CRC '123456789': C=0x{ref_c:04x}, Py=0x{ref_py:04x}, "
          f"{'MATCH' if ref_c == ref_py and ref_c == 0x29B1 else 'MISMATCH'}")
    assert ref_c == ref_py == 0x29B1

    # 2. Random frames: Python encodes, both decoders parse.
    rng = random.Random(0xC0DE)
    failures = 0
    n_trials = 200
    for trial in range(n_trials):
        n = rng.randint(0, MAX_TARGETS)
        targets = [
            Target(
                range_cm   = rng.randint(0, 800),
                speed_cm_s = rng.randint(-1500, 1500),
                angle_cdeg = rng.randint(-9000, 9000),
                rcs        = rng.randint(0, 255),
                flags      = rng.randint(0, 3),
                x_cm       = rng.randint(-800, 800),
                y_cm       = rng.randint(-800, 800),
            ) for _ in range(n)
        ]
        f = Frame(timestamp_ms=rng.randint(0, 0xFFFFFFFF), targets=targets)
        raw = encode_frame(f)

        # Python decode
        f2 = decode_frame(raw)
        if f2 != f:
            failures += 1
            print(f"  trial {trial}: Python decode mismatch")
            continue

        # C decode
        out_targets = (CTarget * MAX_TARGETS)()
        out_ts = ctypes.c_uint32()
        rc = lib.radar_parse_frame(raw, len(raw),
                                   ctypes.cast(out_targets, ctypes.POINTER(CTarget)),
                                   ctypes.byref(out_ts))
        if rc != n:
            failures += 1
            print(f"  trial {trial}: C decode n={rc}, expected {n}")
            continue
        if out_ts.value != f.timestamp_ms:
            failures += 1
            print(f"  trial {trial}: C ts mismatch")
            continue
        for i in range(n):
            t_c = out_targets[i]
            t_p = targets[i]
            if (t_c.range_cm != t_p.range_cm or
                t_c.rcs != t_p.rcs or
                t_c.flags != t_p.flags):
                failures += 1
                print(f"  trial {trial} t[{i}]: uint field mismatch")
            if (t_c.speed_cm_s != t_p.speed_cm_s or
                t_c.angle_cdeg != t_p.angle_cdeg or
                t_c.x_cm != t_p.x_cm or
                t_c.y_cm != t_p.y_cm):
                failures += 1
                print(f"  trial {trial} t[{i}]: int field mismatch")

    # 3. CRC tamper detection (C side)
    raw = encode_frame(Frame(timestamp_ms=999, targets=[
        Target(range_cm=100, speed_cm_s=10)
    ]))
    tampered = bytearray(raw)
    tampered[-1] ^= 0xFF
    out_targets = (CTarget * MAX_TARGETS)()
    out_ts = ctypes.c_uint32()
    rc = lib.radar_parse_frame(bytes(tampered), len(tampered),
                               ctypes.cast(out_targets, ctypes.POINTER(CTarget)),
                               ctypes.byref(out_ts))
    if rc != -1:
        failures += 1
        print(f"  CRC tamper not detected by C (rc={rc})")

    # 4. Magic-byte tamper detection (C side)
    tampered = bytearray(raw)
    tampered[0] = 0x00
    rc = lib.radar_parse_frame(bytes(tampered), len(tampered),
                               ctypes.cast(out_targets, ctypes.POINTER(CTarget)),
                               ctypes.byref(out_ts))
    if rc != -1:
        failures += 1
        print(f"  Magic tamper not detected by C (rc={rc})")

    print(f"\nResult: {n_trials - failures}/{n_trials} cross-language trials passed.")
    if failures == 0:
        print("All C/Python protocol agreement tests passed.")
        return 0
    print(f"{failures} failures.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
