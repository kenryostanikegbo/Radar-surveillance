"""
RD-03D wire-format encoder/decoder.

The firmware (ESP-IDF, C) and the PC simulator (Python) must agree on the
exact byte layout. This file is the canonical source-of-truth for the
wire format. The C side mirrors it -- see firmware/main/radar_protocol.h.

Frame layout (little-endian):
    offset  size  field          type    notes
    ------  ----  -------------  ------  ----------------------------------
         0     2  magic          u16     0xAA55
         2     1  version        u8      0x01
         3     1  n_targets      u8      0..3
         4     4  timestamp_ms   u32     since radar boot
         8     8  reserved       u8[8]   zero-filled
        16    16  target[0]      struct  (see below)
        32    16  target[1]      struct  (optional)
        48    16  target[2]      struct  (optional)
    16+16N    2  crc16          u16     CRC-16/CCITT-FALSE over bytes [0, 16+16N)

Target struct (16 bytes):
    offset  size  field          type    notes
    ------  ----  -------------  ------  ----------------------------------
         0     2  range_cm       u16     0..800 cm (8 m RD-03D max)
         2     2  speed_cm_s     i16     signed, cm/s; -1500..1500
         4     2  angle_cdeg     i16     signed, centi-degrees; -9000..9000
         6     1  rcs            u8      0..255 dBm placeholder
         7     1  flags          u8      bit0: valid, bit1: moving
         8     2  x_cm           i16     Cartesian
        10     2  y_cm           i16
        12     2  z_cm           i16     reserved for now (0)
        14     2  reserved       u16     0

Total: 18 + 16*N bytes. N=3 max -> 66 bytes.

Usage:
    from protocol_encode import encode_frame, decode_frame, Frame, Target

    f = Frame(
        timestamp_ms=1234,
        targets=[
            Target(range_cm=150, speed_cm_s=20, x_cm=140, y_cm=50),
            Target(range_cm=300, speed_cm_s=720, x_cm=-300, y_cm=20),
        ],
    )
    raw = encode_frame(f)
    f2  = decode_frame(raw)
    assert f == f2

    # Self-test (round-trip + CRC correctness):
    python scripts/protocol_encode.py --self-test
"""

import argparse
import struct
from dataclasses import dataclass, field
from typing import List, Optional


# ---- Constants (must match firmware/main/radar_protocol.h) ----
MAGIC          = 0xAA55
VERSION        = 0x01
MAX_TARGETS    = 3
HEADER_SIZE    = 16
TARGET_SIZE    = 16
CRC_SIZE       = 2


# ---- CRC-16/CCITT-FALSE ----
# Polynomial 0x1021, init 0xFFFF, no reflection, no xor-out.
# Same algorithm used by ESP-IDF's esp_rom_crc16_le and by most UART
# sensor protocols. The firmware C side uses the identical table-free
# bit-by-bit implementation in radar_protocol.h:radar_crc16().
def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= (b << 8) & 0xFFFF
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


# ---- Data classes ----
@dataclass
class Target:
    range_cm:    int = 0     # 0..800 cm
    speed_cm_s:  int = 0     # -1500..1500 cm/s
    angle_cdeg:  int = 0     # -9000..9000 centi-degrees
    rcs:         int = 0     # 0..255
    flags:       int = 1     # bit0: valid (default 1)
    x_cm:        int = 0     # signed Cartesian
    y_cm:        int = 0
    z_cm:        int = 0
    _reserved:   int = 0

    def pack(self) -> bytes:
        # Format: range_cm(u16) speed_cm_s(i16) angle_cdeg(i16)
        #         rcs(u8) flags(u8) x_cm(i16) y_cm(i16) z_cm(i16) reserved(u16)
        #         -> 2+2+2+1+1+2+2+2+2 = 16 bytes
        return struct.pack(
            "<HhhBBhhhH",
            self.range_cm & 0xFFFF,
            max(-32768, min(32767, int(self.speed_cm_s))),
            max(-32768, min(32767, int(self.angle_cdeg))),
            self.rcs & 0xFF,
            self.flags & 0xFF,
            max(-32768, min(32767, int(self.x_cm))),
            max(-32768, min(32767, int(self.y_cm))),
            max(-32768, min(32767, int(self.z_cm))),
            self._reserved & 0xFFFF,
        )

    @classmethod
    def unpack(cls, buf: bytes) -> "Target":
        if len(buf) != TARGET_SIZE:
            raise ValueError(f"target buffer must be {TARGET_SIZE} bytes, got {len(buf)}")
        range_cm, speed, angle, rcs, flags, x, y, z, _ = struct.unpack("<HhhBBhhhH", buf)
        return cls(range_cm=range_cm, speed_cm_s=speed, angle_cdeg=angle,
                   rcs=rcs, flags=flags, x_cm=x, y_cm=y, z_cm=z)


@dataclass
class Frame:
    timestamp_ms: int = 0
    targets: List[Target] = field(default_factory=list)

    def __post_init__(self):
        if len(self.targets) > MAX_TARGETS:
            raise ValueError(f"frame can hold at most {MAX_TARGETS} targets, got {len(self.targets)}")

    def pack(self) -> bytes:
        n = len(self.targets)
        if n > MAX_TARGETS:
            raise ValueError(f"too many targets: {n}")
        # Header: magic(u16) version(u8) n_targets(u8) reserved(u8)
        #         timestamp_ms(u32) reserved(u8[7])  -> 2+1+1+1+4+7 = 16 bytes
        header = struct.pack(
            "<HBBBI",
            MAGIC,
            VERSION,
            n,
            0,                                       # reserved byte
            self.timestamp_ms & 0xFFFFFFFF,
        )
        header += b"\x00" * 7                        # 7 reserved bytes -> total 16
        body = b"".join(t.pack() for t in self.targets)
        pad  = b"\x00" * (TARGET_SIZE * (MAX_TARGETS - n))
        payload = header + body + pad
        crc = crc16_ccitt(payload)
        return payload + struct.pack("<H", crc)

    @classmethod
    def unpack(cls, raw: bytes) -> "Frame":
        """Decode a single frame from raw bytes. Validates magic + CRC."""
        if len(raw) < HEADER_SIZE + CRC_SIZE:
            raise ValueError(f"frame too short: {len(raw)} bytes")
        # magic(u16) version(u8) n(u8) reserved(u8) timestamp_ms(u32) -- 8 bytes consumed
        magic, version, n, _r0, ts_ms = struct.unpack_from("<HBBBI", raw, 0)
        if magic != MAGIC:
            raise ValueError(f"bad magic: 0x{magic:04x}, expected 0x{MAGIC:04x}")
        if version != VERSION:
            raise ValueError(f"unsupported version: {version}")
        if n > MAX_TARGETS:
            raise ValueError(f"bad n_targets: {n}")
        body_start = HEADER_SIZE
        body_end = body_start + TARGET_SIZE * MAX_TARGETS
        body = raw[body_start:body_end]
        expected_crc = struct.unpack_from("<H", raw, body_end)[0]
        actual_crc = crc16_ccitt(raw[:body_end])
        if expected_crc != actual_crc:
            raise ValueError(f"bad CRC: got 0x{expected_crc:04x}, expected 0x{actual_crc:04x}")
        targets = [Target.unpack(body[i*TARGET_SIZE:(i+1)*TARGET_SIZE]) for i in range(n)]
        return cls(timestamp_ms=ts_ms, targets=targets)


def encode_frame(frame: Frame) -> bytes:
    """Convenience: pack a Frame to wire bytes."""
    return frame.pack()


def decode_frame(raw: bytes) -> Frame:
    """Convenience: unpack wire bytes to a Frame."""
    return Frame.unpack(raw)


def frame_size(n_targets: int) -> int:
    """Total bytes on the wire for a frame with n_targets targets."""
    return HEADER_SIZE + TARGET_SIZE * MAX_TARGETS + CRC_SIZE


# ---- Self-test ----
def self_test(verbose: bool = False) -> int:
    """Round-trip encode/decode a battery of frames. Returns 0 on success."""
    import random
    rng = random.Random(0xC0DE)

    failures = 0

    # 1. Empty frame
    f = Frame(timestamp_ms=0)
    raw = encode_frame(f)
    f2 = decode_frame(raw)
    if f != f2 or len(raw) != frame_size(0):
        failures += 1
        if verbose: print(f"  FAIL empty: f={f}, f2={f2}, len={len(raw)}")

    # 2. Frame with all targets
    f = Frame(timestamp_ms=12345678, targets=[
        Target(range_cm=150, speed_cm_s=20, angle_cdeg=-4500,
               rcs=42, flags=0x03, x_cm=140, y_cm=50, z_cm=0),
        Target(range_cm=300, speed_cm_s=720, angle_cdeg=2000,
               rcs=80, flags=0x03, x_cm=-300, y_cm=20, z_cm=0),
        Target(range_cm=600, speed_cm_s=-1500, angle_cdeg=9000,
               rcs=10, flags=0x01, x_cm=0, y_cm=-600, z_cm=0),
    ])
    raw = encode_frame(f)
    f2 = decode_frame(raw)
    if f != f2 or len(raw) != frame_size(3):
        failures += 1
        if verbose: print(f"  FAIL full: f={f}, f2={f2}, len={len(raw)}")
    if verbose:
        print(f"  full frame: {len(raw)} bytes, ts={f2.timestamp_ms}, targets={len(f2.targets)}")

    # 3. Random frames
    for trial in range(1000):
        n = rng.randint(0, 3)
        targets = [
            Target(
                range_cm   = rng.randint(0, 800),
                speed_cm_s = rng.randint(-1500, 1500),
                angle_cdeg = rng.randint(-9000, 9000),
                rcs        = rng.randint(0, 255),
                flags      = rng.randint(0, 3),
                x_cm       = rng.randint(-800, 800),
                y_cm       = rng.randint(-800, 800),
                z_cm       = 0,
            )
            for _ in range(n)
        ]
        f = Frame(timestamp_ms=rng.randint(0, 0xFFFFFFFF), targets=targets)
        raw = encode_frame(f)
        f2 = decode_frame(raw)
        if f != f2:
            failures += 1
            if verbose:
                print(f"  FAIL trial {trial}: f={f}, f2={f2}")
            break

    # 4. CRC tampering detection
    f = Frame(timestamp_ms=999, targets=[Target(range_cm=100, speed_cm_s=10)])
    raw = bytearray(encode_frame(f))
    raw[-1] ^= 0xFF  # corrupt CRC
    try:
        decode_frame(bytes(raw))
        failures += 1
        if verbose: print("  FAIL CRC tamper not detected")
    except ValueError:
        pass  # expected

    # 5. Magic-byte detection
    raw = bytearray(encode_frame(f))
    raw[0] = 0x00
    try:
        decode_frame(bytes(raw))
        failures += 1
        if verbose: print("  FAIL magic tamper not detected")
    except ValueError:
        pass

    # 6. CRC reference vector: CRC-16/CCITT-FALSE on ASCII "123456789" = 0x29B1
    ref = crc16_ccitt(b"123456789")
    if ref != 0x29B1:
        failures += 1
        if verbose: print(f"  FAIL CRC reference: got 0x{ref:04x}, expected 0x29B1")

    if verbose:
        if failures == 0:
            print("All protocol self-tests passed.")
        else:
            print(f"{failures} self-test failures.")
    return failures


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--self-test", action="store_true",
                   help="Run round-trip and CRC validation")
    args = p.parse_args()
    if args.self_test:
        rc = self_test(verbose=True)
        return 0 if rc == 0 else 1
    # Demo: print one empty frame and one full frame as hex.
    print("Empty frame:", encode_frame(Frame(timestamp_ms=0)).hex())
    f = Frame(timestamp_ms=1234, targets=[
        Target(range_cm=150, speed_cm_s=20, x_cm=140, y_cm=50),
    ])
    print("1-target    :", encode_frame(f).hex())
    print(f"Frame size  : {frame_size(0)} bytes empty, "
          f"{frame_size(3)} bytes full (max)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
