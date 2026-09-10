# Radar Surveillance System

A 24 GHz FMCW radar (RD-03D) → ESP32-S3 → Wi-Fi dashboard pipeline that
classifies detections as **human**, **vehicle**, or **unknown** in real
time. Built around a quantized INT8 1-D CNN / MLP trained on the
RadHAR dataset, deployed on a $5 microcontroller with no GPU and
~250 KB of RAM.

## What's in this repo

```
.
├── README.md                       (this file)
├── TESTING_STRATEGY.md             3-layer validation plan
├── Radar_Surveillance_System_Project_Synopsis.pdf
└── phase2/                         All code + models
    ├── firmware/                   ESP-IDF firmware (C)
    │   ├── main.c                  4-task FreeRTOS entry
    │   ├── radar_protocol.h        Wire format
    │   ├── parse_frame.c           CRC + magic
    │   ├── feature_extract.c       16-frame window -> 72-dim features
    │   ├── tracker.c               alpha-beta filter
    │   ├── inference.c             TFLite Micro wrapper
    │   ├── dashboard_http.c        JSON serializers
    │   └── main/dashboard/         HTML/JS/CSS (embedded into firmware)
    ├── models/                     Trained MLP + INT8 TFLite + norm stats
    ├── scripts/
    │   ├── compute_features.py     Feature extraction (Python reference)
    │   ├── train_mlp.py            MLP training
    │   ├── train_1dcnn.py          1-D CNN training
    │   ├── quantize_export.py      Keras -> INT8 TFLite + C array
    │   ├── protocol_encode.py      Wire-format encoder (Python)
    │   ├── pc_radar_sim.py         Replay dataset over UART
    │   ├── build_firmware.bat      Compile C modules -> radar_parse.dll
    │   └── ...
    ├── tests/
    │   ├── cross_check_features.py C vs Python feature parity (50 trials)
    │   ├── loopback_test.py        End-to-end host validation
    │   └── dashboard_server.py     Local HTTP server with the real UI
    ├── data/                       Windowed radar datasets (gitignored)
    ├── RUN_LOG.md                  Build/run log
    └── CALIBRATION_REPORT.md       Calibration + noise-stress results
```

## Validation strategy

The radar arrives in Week 9 — but everything else is buildable today.
Three layers, progressing from pure-software to full hardware:

| Layer | Runs on | What it validates |
|-------|---------|-------------------|
| **1 — Offline** | PC + Python | Model accuracy, calibration, noise stress, logit margins |
| **2 — Loopback** | PC + C DLL + Python | Parser, feature extractor, tracker, JSON shape — exact firmware code |
| **3 — Hardware** | ESP32-S3 + RD-03D | UART timing, TFLite Micro, FreeRTOS, Wi-Fi softAP, dashboard polling |

Layer 2 is the bridge: every C module the ESP32-S3 will run is
compiled to `radar_parse.dll` and exercised on the host via `ctypes`.
This catches wire-format drift, feature-extraction drift, integer
overflow, and JSON schema mismatches **months before** the radar is on
the desk. See [TESTING_STRATEGY.md](TESTING_STRATEGY.md) for the full
plan.

## Phase 3 status (2026-09-10)

| Component | State | Evidence |
|-----------|-------|----------|
| Wire protocol (Python ↔ C) | Done, bit-exact | `cross_check_c_python.py`: 200/200 trials |
| Feature extractor (Python ↔ C) | Done, bit-exact within float32 ε | `cross_check_features.py`: 50/50 trials, max diff 3e-3 |
| INT8 model | Done, 99.1% accuracy on clean Test | `noise_stress_test.py`, `CALIBRATION_REPORT.md` |
| Tracker | Done, alpha-beta | `tests/dashboard_server.py` runs end-to-end |
| Inference path | NumPy fallback (TFLite Micro stub) | `tests/loopback_test.py`: 99.6% accuracy |
| Dashboard HTML/JS/CSS | Done, served by host | `tests/dashboard_server.py` |
| FreeRTOS layout | Designed, not yet wired | `firmware/main.c` |
| ESP-IDF build | Pending — needs IDF v5.2+ toolchain | n/a |

## Quickstart (no hardware required)

You can validate the firmware pipeline on any Windows + Python 3.11
machine with MSVC build tools installed:

```bash
cd phase2

# 1. Build the C firmware modules into a Windows DLL
cmd /c scripts\build_firmware.bat
# Output: C:\tmp\radar_parse.dll

# 2. Run the three Phase 3 verifications
python scripts/protocol_encode.py --self-test
# All protocol self-tests passed.

python tests/cross_check_features.py
# 50/50 trials passed, max diff 3.019e-03.

python tests/loopback_test.py --frames 1000 --data data/combined_windows.npz --shuffle
# 1000 frames parsed, 62 windows, 0 failures, PASS.

# 3. Open the live dashboard at http://127.0.0.1:8080
python tests/dashboard_server.py --frames 0 --shuffle
# (auto-opens a browser tab)
```

The dashboard shows the exact UI the ESP32-S3 will serve over its
softAP — radar canvas, range rings, class-colored tracks (green =
human, blue = vehicle, gray = unknown), live track list with per-track
ID / class / confidence / position / speed / age / hits.

## Hardware build (when the radar arrives)

```bash
cd phase2/firmware

# Requires ESP-IDF v5.2+ and the esp-tflite-micro component
idf.py set-target esp32-s3
idf.py build
idf.py flash --port COM7
idf.py monitor

# In another terminal, replay recorded radar data over real UART:
python scripts/pc_radar_sim.py --port COM7 --baud 256000 --fps 20

# Open the dashboard on the softAP:
#   Wi-Fi SSID: RadarSurv-XXXX
#   Password:   radar1234
#   URL:        http://192.168.4.1/
```

## Architecture

```
RD-03D 24 GHz FMCW radar
  │  UART2 @ 256 000 baud, 8N1
  ▼
ESP32-S3 (N8R8, 8 MB PSRAM)
  │
  ├── radar_rx_task    (core 0, prio 10) → UART DMA → window buffer
  ├── inference_task   (core 1, prio 8)  → features → TFLite Micro
  ├── tracker_task     (core 1, prio 5)  → alpha-beta filter
  └── http_server_task (core 0, prio 1)  → softAP + esp_http_server
                                          ↓
                                  Wi-Fi softAP
                                  (RadarSurv-XXXX / radar1234)
                                          ↓
                          Browser dashboard polling /api/state
```

Wire format (66 bytes/frame at 20 Hz → 1 320 B/s, well within the
256 000 baud budget):

```
┌────────┬─────┬─────┬─────┬────────┬───────────┬─────┐
│ magic  │ ver │ n_t │ ts  │  pads  │ n×target  │ crc │
│ u16 LE │ u8  │ u8  │u32LE│ u8[8]  │ 16 bytes  │ u16 │
└────────┴─────┴─────┴─────┴────────┴───────────┴─────┘
```

Target struct (16 bytes): `range_cm`, `speed_cm_s`, `angle_cdeg`,
`rcs`, `flags`, `x_cm`, `y_cm`, `z_cm`, `reserved`.

## Model

- **Architecture**: 72 → 32 (ReLU) → 16 (ReLU) → 3 (softmax).
- **Quantization**: INT8, input scale 0.074184, input ZP -13;
  output scale 0.003906, output ZP -128.
- **Accuracy**: 100% on held-out RadHAR Test split; 99.1% under
  σ=1.0 per-feature Gaussian noise.
- **Size**: 6.6 KB (6792 bytes) in `models/model_int8.tflite`.
- **Feature vector**: 17 per-target stats × 3 targets + 21 global
  stats = 72 floats. Direct port of `scripts/compute_features.py`.

See [RUN_LOG.md](phase2/RUN_LOG.md) and
[CALIBRATION_REPORT.md](phase2/CALIBRATION_REPORT.md) for the
full training history and calibration analysis.

## Repository conventions

- C source uses `#pragma pack(push, 1)` on MSVC, `__attribute__((packed))`
  on GCC — see `firmware/radar_protocol.h`.
- Build artifacts (`*.dll`, `*.obj`, `*.lib`, `*.exp`) go to `C:\tmp\`
  on Windows; they're rebuilt by `scripts/build_firmware.bat`.
- `data/` is gitignored — datasets are too large for version control.
  Use `scripts/run_phase2_pipeline.py` to regenerate from raw sources.
- `models/model_int8.tflite` and `models/model_metadata.h` are
  committed (small, regenerated by `quantize_export.py`).

## License

TBD — academic project, not yet licensed for redistribution.
