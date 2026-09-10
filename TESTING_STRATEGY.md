# Testing Strategy: 3-Layer Validation

**Project**: Radar Surveillance System with Embedded AI
**Target hardware**: RD-03D 24 GHz FMCW radar + ESP32-S3
**Constraint**: RD-03D arrives after Phase 2 model development is complete

---

## Problem Statement

The full system (radar → ESP32-S3 → AI classification → dashboard) cannot be tested end-to-end before the RD-03D arrives. Without a validation strategy, we risk discovering hardware integration issues late in the timeline.

**Solution**: Three-layer validation that progressively exercises the system from pure-software to full hardware integration.

---

## Layer 1: Off-Device Model Validation (PC only)

### When
Phase 2, Weeks 4–6 — runs in parallel with dataset preparation and model training.

### What is tested
- Model accuracy on the ICLegend-derived dataset
- INT8 quantization correctness (PC TFLite runtime executes the same quantized model that will run on ESP32-S3)
- Latency on PC (sanity check, not representative of ESP32-S3 timing)

### Method
Run `test_model_offline.py` to load `model_int8.tflite`, iterate over the test split, and compute accuracy / per-class F1 / latency.

### Pass criteria
- Overall F1-score ≥ 0.80
- Human class F1 ≥ 0.85
- Vehicle class F1 ≥ 0.80
- Unknown class F1 ≥ 0.70
- Model file size ≤ 50 KB

### What this does NOT prove
- Model actually runs on ESP32-S3 hardware
- Memory footprint fits in ESP32-S3 RAM
- Real-world inference timing
- UART parsing works
- FreeRTOS scheduling is correct

---

## Layer 2: ESP32-S3 Dry Run with UART Simulator (PC → ESP32-S3)

### When
Phase 3, Weeks 7–8 — after the model is trained and exported, before RD-03D arrives.

### What is tested
- ESP-IDF UART parser correctly decodes RD-03D-format binary frames
- FreeRTOS tasks schedule properly (radar reader, inference, tracker, dashboard server)
- TFLite / ESP-DL model actually loads and runs on real ESP32-S3 hardware
- Inference latency on real ESP32-S3 (target: ≤ 100 ms per window)
- RAM and flash footprint fit within ESP32-S3 limits
- Tracker maintains ID across simulated frames
- Dashboard receives and renders data over Wi-Fi softAP
- Power consumption on real hardware
- 24-hour soak test for memory leaks / crashes

### Method
A Python script on the PC (`pc_radar_sim.py`) reads the ICLegend-derived dataset and streams it out a real serial port (USB-to-UART adapter) at 256,000 baud in the RD-03D binary frame format. The ESP32-S3 receives these frames exactly as if they came from the radar.

```
┌─────────────┐  UART 256000 baud  ┌──────────┐    Wi-Fi   ┌──────────┐
│  PC running │ ─────────────────► │ ESP32-S3 │ ─────────► │ Dashboard│
│  radar sim  │  real serial port  │          │            │          │
└─────────────┘                    └──────────┘            └──────────┘
       │
       └── streams pre-recorded RD-03D-format frames
            from the ICLegend dataset (rd03d_dataset.npz)
```

### Hardware required
- ESP32-S3-DevKitC-1 (N8R8 variant, 8 MB PSRAM)
- USB-to-UART adapter (CP2102 or CH340) — connected to a free COM port on the PC
- Breadboard + jumper wires
- Optional: OLED 0.96" I2C for local status display
- Optional: USB multimeter for power measurement
- PC with Python 3.10+ and `pyserial` installed

### Wiring
| ESP32-S3 GPIO | USB-UART adapter |
|---|---|
| GPIO 17 (RX) | TX |
| GPIO 18 (TX) | RX (unused for sim, only need one-way) |
| GND | GND |
| 5V or 3V3 | (do not power adapter from ESP32 — use separate USB) |

### Pass criteria
- 100% of frames parsed without error
- Inference latency on ESP32-S3 ≤ 100 ms (measured with `esp_timer_get_time()`)
- RAM peak usage ≤ 250 KB (`heap_caps_get_min_free_heap(MALLOC_CAP_8BIT)`)
- Flash usage ≤ 1.5 MB (firmware + model)
- Power consumption ≤ 0.5 W average
- 24-hour soak test: zero crashes, zero memory leaks (free heap stable within ±2 KB)
- Dashboard updates at ≥ 5 Hz

### What this does NOT prove
- Real radar RF behaviour
- Real-world multipath / clutter signatures
- Real human / vehicle signatures (only ICLegend's test subjects and synthetic data)
- Hardware integration with RD-03D's specific quirks

---

## Layer 3: Hardware Integration (RD-03D live)

### When
Phase 3, Weeks 9–10 — once RD-03D arrives.

### What is tested
- Real radar RF chain produces expected target reports
- ESP32-S3 parses real RD-03D output without modification
- Classification accuracy on real signatures vs ground truth
- Tracking performance in real multipath environment
- Behaviour in low-visibility conditions (fog, smoke, night)
- Range limitation honestly measured and documented

### Method
Disconnect USB-UART simulator. Connect RD-03D to same ESP32-S3 UART pins. Same firmware, same model, same dashboard. Run controlled tests with:

| Scenario | Ground truth method |
|---|---|
| Human walking at 2 m / 4 m / 6 m | Stopwatch + tape measure |
| Human standing still | Visual |
| Vehicle (bicycle / car) passing | Phone camera timestamp |
| Empty scene | Confirmed by camera |
| Smoke / fog interference | Fog machine, smoke emitter |
| Night | Dark room / after dusk |

### Pass criteria
- F1-score on real signatures within 15% of Layer 1 result (some drop expected)
- Detection of human up to RD-03D's actual max range (typically 8 m)
- Tracker maintains ID through brief occlusions (1–2 frames)
- Dashboard reflects reality within 200 ms

### Recalibration pass
When RD-03D arrives, plan a 1–2 day recalibration:
1. Collect 100–200 live samples with ground truth labels
2. Fine-tune the trained model on this small calibration set (transfer learning, freeze most layers, retrain final dense layer)
3. Re-export to INT8 TFLite
4. Re-deploy to ESP32-S3
5. Re-run Layer 3 tests

---

## Acknowledged Limitations

1. **Domain gap**: ICLegend was collected in their lab with their specific antenna patterns and radar chip. RD-03D has different beam patterns and noise floors. Expect F1 to drop 5–15% when switching hardware.

2. **Multi-target edge cases**: The simulator sends clean, well-formed frames. RD-03D will occasionally send malformed frames, missed frames, or stuck-at-zero reports. Mitigation: UART watchdog that clears all tracks if no valid frame received in 200 ms.

3. **People vs. lab subjects**: ICLegend dataset has specific test subjects. Real humans vary in height, gait, body shape. Mitigation: add real calibration data during the recalibration pass.

4. **100–200 m requirement not addressed**: This is future work per the project synopsis. RD-03D detection range is 0.5–8 m. Long-range operation requires 77 GHz radar and a more capable processor.

---

## Cost Summary for Layer 2 Validation

| Item | Cost (₦) | Purpose |
|---|---|---|
| ESP32-S3-DevKitC-1 N8R8 | 18,000 | Inference target |
| USB-to-UART cable (CP2102/CH340) | 5,000 | Connect PC to ESP32-S3 |
| Breadboard + jumper wires | 5,000 | Quick wiring |
| MicroSD module (SPI) | 5,000 | Log classifications for analysis |
| OLED 0.96" I2C | 3,500 | Local status display |
| USB multimeter (optional) | 5,000 | Verify power consumption |
| **Total** | **~41,500** | |

Does not require the RD-03D.

---

## Timeline Summary

| Week | Activity | Layer |
|---|---|---|
| 4 | Dataset conversion | — |
| 5 | Synthetic data + EDA | — |
| 6 | Model training, INT8 export | Layer 1 (off-device) |
| 7 | UART simulator + ESP-IDF parser skeleton | Layer 2 build-up |
| 8 | Full firmware integration, 24h soak test | Layer 2 validation |
| 9 | RD-03D arrives, recalibration pass | Layer 3 |
| 10 | Live field testing | Layer 3 |

---

## Why This Strategy Works

- **Layer 1 catches AI bugs** (bad architecture, broken preprocessing, class imbalance)
- **Layer 2 catches firmware bugs** (UART parsing, FreeRTOS, memory, Wi-Fi, dashboard)
- **Layer 3 catches hardware bugs** (real radar signature mismatch, multipath, range limits)

By the time the RD-03D arrives, every other component has been independently validated, so hardware integration becomes a focused 1–2 day task rather than a debugging nightmare.
