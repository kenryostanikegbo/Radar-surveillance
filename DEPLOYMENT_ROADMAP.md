# RD-03D + ESP32-S3 AI Radar Surveillance System — Deployment Roadmap

**Project phase:** Transition from trained ML model and PC-based radar simulation to live RD-03D acquisition, ESP32-S3 inference, physical LCD output, alerts, and logging.  
**Status:** Hardware deployment and end-to-end validation pending.  
**Last updated:** 2026-11-17

> [!IMPORTANT]
> The saved offline model metrics are not proof of accuracy on physical RD-03D measurements. Before claiming successful deployment, validate the complete pipeline with real sensor captures.

## 1. Starting point

### Already available

- [x] ML training code and trained MLP / 1D-CNN models.
- [x] Exported INT8 TFLite model and C model data (`phase2/models/model_int8.tflite`, `model_data.cc`).
- [x] Saved feature/normalization metadata (`model_metadata.h`, `norm_params.npz`).
- [x] Radar simulation tools and PC dashboard.
- [x] Feature extraction code, tracking framework, and test scripts.
- [x] ESP32-S3 firmware framework, including FreeRTOS task design and browser-dashboard assets.

### Not yet established by the supplied implementation

- [ ] Live RD-03D UART capture and decoding on ESP32-S3.
- [ ] Working TensorFlow Lite Micro inference on the microcontroller.
- [ ] Validated inference using real radar captures.
- [ ] LCD, buzzer, LED, and microSD drivers integrated and tested.
- [ ] End-to-end standalone hardware test.

**Key architecture:**

```text
RD-03D radar (UART)
      |
      v
ESP32-S3 receive + native frame parser
      |
      v
16-frame window (up to 3 target slots; 4 measurements per slot)
      |
      v
72-feature extraction -> training-matched normalization -> INT8 quantization
      |
      v
TFLite Micro inference (human / vehicle / unknown)
      |
      +--> Target tracking and decision/alert logic
                |        |         |
                v        v         v
              LCD     Buzzer/LED   Wi-Fi dashboard + microSD event log
```

**Design caution:** A *no-target* state must be handled separately from the model's *unknown* class. The current model predicts one label per window, not necessarily one independent label for each simultaneous target.

## 2. Deployment stages

Estimated durations below are planning ranges, not commitments. Complete each stage's exit criterion before relying on the next stage.

### Stage 1 — Establish the ESP32-S3 firmware baseline

**Estimate:** 1–3 days  
**Goal:** Build, flash, and debug a minimal firmware baseline before integrating peripherals.

- [ ] Install the required ESP-IDF toolchain and select the `esp32s3` target.
- [ ] Confirm exact board model, available flash/PSRAM, supply pins, and safe GPIO assignments.
- [ ] Verify and correct the ESP-IDF project/component layout and `CMakeLists.txt` as needed.
- [ ] Confirm `sdkconfig.defaults`, `partitions.csv`, and model-related source files are included correctly.
- [ ] Build and flash a minimal program; verify serial-monitor output.
- [ ] Record toolchain version and reproducible build/flash commands in `README.md`.

**Exit criterion:** A teammate can clone the repository, build and flash the firmware, and observe successful startup logging.

**Relevant files:** `phase2/firmware/CMakeLists.txt`, `main.c`, `sdkconfig.defaults`, `partitions.csv`, `phase2/scripts/build_firmware.bat`.

### Stage 2 — Connect and validate the RD-03D radar

**Estimate:** 2–4 days  
**Goal:** Read real measurements from the sensor, independently of ML.

- [ ] Connect RD-03D to a Windows PC using a suitable USB–UART interface and verify it using the vendor's visualization tool.
- [ ] Confirm the sensor's supply requirements, UART logic levels, baud rate, packet layout, and coordinate units against the exact hardware documentation.
- [ ] Wire RD-03D to ESP32-S3 UART with common ground and compatible electrical levels.
- [ ] Capture and preserve representative raw UART frames (no target, one target, multiple targets where supported).
- [ ] Replace the stub in `radar_uart.c` with working UART initialization and reads.
- [ ] Implement and test decoding of **native RD-03D packets**; do not assume the simulator's custom packet format matches the sensor.
- [ ] Show target count, coordinates, distance, and motion measurements on the serial monitor.
- [ ] Add timeout, malformed-frame, and reconnection handling.

**Exit criterion:** Moving in front of the sensor causes plausible, continuous target updates in ESP32-S3 serial logs, with no ML involved.

**Relevant files:** `phase2/firmware/radar_uart.c`, `radar_uart.h`, `parse_frame.c`, `radar_protocol.h`, `phase2/scripts/protocol_encode.py` (simulator reference only).

### Stage 3 — Deploy the trained INT8 model

**Estimate:** 3–6 days  
**Goal:** Run the existing trained model correctly on the ESP32-S3.

- [ ] Integrate an ESP-compatible TensorFlow Lite Micro runtime and document its version.
- [ ] Include the exported INT8 model (`model_data.cc`) and validate its tensor types/shapes.
- [ ] Register required TFLM operators and allocate a tensor arena appropriate for the board.
- [ ] Replace the placeholder `inference_run()` implementation with actual interpreter invocation.
- [ ] Reuse the 16-frame buffer and 72-feature C extraction pipeline.
- [ ] Apply exactly the saved training normalization and quantization parameters.
- [ ] Compare C/ESP32 feature vectors and model predictions with the Python implementation on **identical recorded windows**.
- [ ] Measure memory use, runtime latency, and confidence output handling.

**Exit criterion:** The ESP32 runs the real model and produces predictions consistent with the Python reference on identical inputs; performance measurements are recorded.

**Relevant files:** `phase2/firmware/inference.c`, `inference.h`, `feature_extract.c`, `feature_extract.h`, `phase2/models/model_int8.tflite`, `model_data.cc`, `model_metadata.h`, `norm_params.npz`, `phase2/tests/cross_check_features.py`.

### Stage 4 — Run inference with live RD-03D data

**Estimate:** 3–7 days  
**Goal:** Replace simulated inputs with physical radar measurements and determine whether the model generalizes.

- [ ] Feed native UART-decoded measurements into the model's 16-frame window.
- [ ] Keep target slots stable across frames as much as the sensor data allows; handle empty targets explicitly.
- [ ] Check timestamps, frame rate, coordinate conventions, units, and missing-data behavior.
- [ ] Implement a separate no-target state before running/displaying classification.
- [ ] Stream real classification label and confidence to the serial monitor.
- [ ] Record labeled live examples (human, vehicle, unknown) where safely and practically possible.
- [ ] Report confusion matrix, false alarms, and latency using real captures.
- [ ] If real-world results are inadequate, correct preprocessing and/or collect additional real training data before retraining.
- [ ] Decide whether scene-level classification suffices or per-target classification is required.

**Exit criterion:** The ESP32 classifies live observations and the team has documented real-world performance and known failure cases.

**Relevant files:** `phase2/firmware/main.c`, `radar_uart.c`, `feature_extract.c`, `inference.c`, `tracker.c`, `phase2/scripts/test_model_offline.py`.

### Stage 5 — Integrate LCD and local alerts

**Estimate:** 2–4 days  
**Goal:** Make essential results visible without a PC.

- [ ] Confirm the actual LCD model; the uploaded schematic depicts a character LCD via PCF8574 I²C expander.
- [ ] Verify LCD and I²C voltage compatibility with ESP32-S3 (use level shifting when required).
- [ ] Select pins that do not conflict with radar UART, boot, PSRAM, or other peripherals.
- [ ] Implement/test an LCD driver and print startup and no-target messages.
- [ ] Show class, distance, and optionally confidence or target count when a target is detected.
- [ ] Implement LED and buzzer GPIO/switching control with safe drivers and power limits.
- [ ] Add alert threshold, cooldown/hysteresis, and false-alarm mitigation.
- [ ] Ensure LCD updates and alert outputs do not block UART processing or inference.

**Exit criterion:** The LCD correctly reflects live state and configured alerts activate reliably without a PC.

**Relevant files:** New display and alert driver files under `phase2/firmware/`, plus `main.c`.

### Stage 6 — Integrate dashboard, Wi-Fi, and microSD logging

**Estimate:** 2–5 days  
**Goal:** Use the existing GUI with live ESP32 data and preserve detections.

- [ ] Finish/validate the embedded HTTP server and dashboard asset serving.
- [ ] Connect the dashboard's `/api/state` polling to real tracked detections and model outputs.
- [ ] Remove or clearly distinguish simulated data from live data.
- [ ] Configure Wi-Fi mode, access controls, and credential handling safely.
- [ ] Implement microSD initialization and event logging with fault handling.
- [ ] Record event timestamp, class, confidence, target position, and alert state as available.
- [ ] Verify that Wi-Fi and SD writes do not disrupt radar reception.
- [ ] Test behavior when Wi-Fi disconnects or the SD card is absent.

**Exit criterion:** A phone/laptop can view live detections and the device continues working and logging under normal operating conditions.

**Relevant files:** `phase2/firmware/dashboard_http.c`, `dashboard_http.h`, `main/dashboard/index.html`, `app.js`, `style.css`, `phase2/tests/dashboard_server.py`.

### Stage 7 — End-to-end field validation and handoff

**Estimate:** 3–5 days  
**Goal:** Demonstrate and document a reliable standalone device.

- [ ] Test no target, one target, and multiple target scenarios.
- [ ] Test different positions, distances, speeds, and motion directions.
- [ ] Measure end-to-end detection latency and inference rate.
- [ ] Evaluate live classification accuracy, false-positive rate, and missed detections.
- [ ] Test continuous operation, restart/recovery behavior, and memory stability.
- [ ] Verify power integrity, cable connections, safe switching drivers, and enclosure.
- [ ] Test disconnected Wi-Fi and missing/corrupted SD-card scenarios.
- [ ] Save a reproducible test procedure, logs, metrics, wiring diagram, and demonstration video.
- [ ] Update `PROJECT_STATUS.md`, `README.md`, and the final report with **measured** results.

**Exit criterion:** A teammate can reproduce the setup, demonstrate live detection and inference, and explain its performance and limitations.

## 3. Recommended implementation order

### First hardware milestone (do this before LCD integration)

```text
RD-03D -> ESP32-S3 UART -> Native packet decoding -> Serial monitor
```

A successful test should show real target measurements changing when a person moves through the sensor's detection area.

### First ML milestone

```text
RD-03D -> 16-frame buffer -> 72 features -> INT8 model -> Serial classification
```

Compare predictions with Python on the same saved sensor windows before relying on live classifications.

### Final integrated milestone

```text
RD-03D -> ESP32-S3 -> INT8 inference + tracking
                         |      |       |
                         v      v       v
                        LCD   Alerts   Wi-Fi dashboard + microSD
```

## 4. Existing project files: ownership guide

| Area | Starting files | Expected work |
| --- | --- | --- |
| Firmware orchestration | `phase2/firmware/main.c` | Initialize components and coordinate tasks |
| Radar acquisition | `radar_uart.c`, `radar_uart.h` | Replace UART placeholder and handle sensor disconnects |
| Frame parsing | `parse_frame.c`, `radar_protocol.h` | Decode actual RD-03D protocol |
| Features | `feature_extract.c`, `.h` | Reuse and cross-check against Python |
| Embedded inference | `inference.c`, `inference.h` | Replace inference placeholder with TFLM |
| Model | `phase2/models/model_data.cc`, `model_metadata.h` | Embed model and apply correct preprocessing |
| Tracking | `tracker.c`, `tracker.h` | Validate timing and target association |
| Dashboard | `dashboard_http.c`, `main/dashboard/` | Connect to live `/api/state` output |
| Offline simulation | `phase2/scripts/`, `phase2/tests/` | Regression tests and reproducibility |
| Hardware outputs | New LCD/buzzer/LED/microSD modules | Implement and test peripheral drivers |

## 5. Important technical risks and decisions

1. **Native packet format:** Simulator packets are not evidence of the RD-03D's actual UART wire protocol; validate with real captures and documentation.
2. **Model generalization:** Offline reported accuracy comes from converted/synthetic data; physical RD-03D performance remains unverified.
3. **Input compatibility:** Confirm that the sensor exposes the measurements required by the model's 16 × 3 × 4 representation; otherwise adapt the pipeline with explicit validation.
4. **Scene vs. per-target class:** The model outputs one label for a window. Avoid assigning it to every simultaneous target without a justified design.
5. **Timing:** Capture, windowing, inference, display updates, HTTP, and SD writes must coexist without dropped frames.
6. **ESP32 variant mismatch:** The uploaded circuit diagram uses a generic ESP32 DevKit while the firmware is intended for ESP32-S3. Verify final wiring, pins and voltage levels.
7. **Power and safety:** Do not drive high-current loads directly from an ESP32 GPIO. Validate switching circuits and power supply before connecting peripherals.

## 6. Team collaboration and handoff

Recommended Git feature branches:

- `feature/radar-uart`
- `feature/tflite-inference`
- `feature/lcd-display`
- `feature/live-dashboard`
- `feature/sd-logging`

For each pull request, include: **what changed**, **how to reproduce the test**, **expected output**, **hardware used**, and **remaining limitations**.

Keep the repository focused on source, build files, deployment model, tests, and documentation. Store large raw training data separately and document how teammates can obtain it. Do not commit passwords or Wi-Fi credentials.

### Handoff checklist

- [ ] `README.md` has installation, build, flash, and test commands.
- [ ] `PROJECT_STATUS.md` reflects current completed and blocked tasks.
- [ ] Hardware wiring and GPIO assignments are documented.
- [ ] Every model file has its corresponding preprocessing metadata.
- [ ] Sample sensor logs and expected outputs are available.
- [ ] The next owner can identify and run the next unfinished stage.

---

**Definition of done:** The system works with **real RD-03D measurements**, executes the **trained INT8 model on ESP32-S3**, provides **independent LCD/alert output**, supports the **live dashboard and event logging**, and has **documented, reproducible real-world tests**.
