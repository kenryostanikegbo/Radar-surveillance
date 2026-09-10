# ESP32-S3 Firmware for Radar Surveillance System

ESP-IDF firmware skeleton for Layer 2 validation. RD-03D radar hasn't
arrived yet, so the firmware's parser / feature / tracker / dashboard
logic is exercised today via the host-side loopback harness.

## Status

| Module                  | State                            |
|-------------------------|----------------------------------|
| `radar_protocol.h`      | Done                             |
| `parse_frame.c`         | Done — bit-exact with Python     |
| `feature_extract.c`     | Done — 50/50 trials, max 3e-3    |
| `tracker.c`             | Done — alpha-beta filter         |
| `inference.c`           | TFLite Micro wrapper + `int_softmax()` |
| `dashboard_http.c`      | Done — JSON serializers          |
| `dashboard/index.html`  | Done                             |
| `dashboard/app.js`      | Done — canvas radar plot         |
| `dashboard/style.css`   | Done                             |
| `main.c`                | Done — 4-task FreeRTOS layout    |
| `CMakeLists.txt`        | Done                             |
| `sdkconfig.defaults`    | Done                             |
| `partitions.csv`        | Done                             |

## Target

- **MCU**: ESP32-S3-DevKitC-1, N8R8 (8 MB PSRAM)
- **Baud**: 256 000 8N1 over UART2
- **Flash budget**: 1.5 MB (firmware + 6.6 KB INT8 model + assets)
- **RAM budget**: 250 KB peak
- **Inference budget**: 100 ms / window (target)

## Layout

```
firmware/
  CMakeLists.txt         ESP-IDF project root
  sdkconfig.defaults     ESP32-S3 + PSRAM + softAP
  partitions.csv         single-app 1.5 MB
  radar_protocol.h       Wire format structs, CRC
  parse_frame.c          CRC + magic validation; bit-exact with Python
  radar_uart.{h,c}       UART2 driver (DMA + ring buffer)
  feature_extract.{h,c}  16-frame window -> 72-dim features
  tracker.{h,c}          alpha-beta tracker (no matrix inversion)
  inference.{h,c}        TFLite Micro wrapper (calls int_softmax)
  dashboard_http.{h,c}   JSON serializers for /api/state
  main.c                 App entry; 4 FreeRTOS tasks
  main/dashboard/
    index.html           Polling SPA (vanilla JS, no framework)
    app.js               200 ms poll, canvas plot
    style.css            dark theme
  test_tracker.c         Host-side unit test
  README.md              this file
```

## Task layout (main.c)

```
core 0                                core 1
────────────────────────────────      ────────────────────────────────
radar_rx_task  (prio 10)              inference_task (prio 8)
  UART2 DMA -> window queue             window_q -> features -> TFLite
                                         -> classify_q

http_server_task (prio 1)             tracker_task   (prio 5)
  softAP + esp_http_server              classify_q -> alpha-beta tracks
```

Queues are bounded; if downstream is slow we drop oldest and log a
warning rather than block upstream radar RX.

## Build prerequisites

```
ESP-IDF v5.2+ (idf.py available in PATH).
Component registry: espressif/esp-tflite-micro.
```

## Build & flash (when IDF is installed)

```bash
cd firmware
idf.py set-target esp32-s3
idf.py build
idf.py flash --port COM7
idf.py monitor
```

## Verify without hardware (today)

```bash
cd phase2

# 1. Build the host-side C DLL (Windows + MSVC build tools required)
cmd /c scripts\build_firmware.bat

# 2. Protocol round-trip self-test
python scripts/protocol_encode.py --self-test

# 3. Feature parity: C vs Python on 50 random windows
python tests/cross_check_features.py

# 4. End-to-end loopback: simulator -> C parser -> C features ->
#    real MLP inference -> Python tracker -> /api/state JSON.
#    Shuffle so all 3 classes get exercised.
python tests/loopback_test.py --frames 8000 --data data/combined_windows.npz --shuffle
```

Pass criteria:

- `protocol_encode.py --self-test` exits 0
- `cross_check_features.py` reports 50/50 trials passed
- `loopback_test.py` reports 0 parse / feature failures and a
  classification accuracy matching the offline calibration (≥99% on
  the combined windows). `PASS: loopback pipeline end-to-end successful`.

## When the radar arrives

```bash
# Replay the recorded dataset over the real UART to ESP32-S3
python scripts/pc_radar_sim.py --port COM7 --baud 256000 --fps 20

# Or, in another terminal, hit the dashboard:
curl http://192.168.4.1/api/state
```

The dashboard polls `/api/state` every 200 ms and renders tracks on a
canvas top-down radar plot with class-colored markers.
