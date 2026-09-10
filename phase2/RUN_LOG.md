# Phase 2 Run Log

**Date**: 2026-09-09
**Environment**: Python 3.11.9 in `venv/` (TensorFlow 2.21.0, numpy 2.4.6, scikit-learn 1.9.0)
**Data source**: Real RadHAR (humans) + synthetic (vehicles, unknown) = 6000 windows
**Hardware**: CPU only (TF GPU not available on native Windows)

---

## Dataset summary

| Class    | Samples | Source                                                |
|----------|---------|-------------------------------------------------------|
| human    | 3000    | `data/iclegend_raw/RadHAR/Data/Train/` (real, 24 GHz) |
| vehicle  | 1500    | `generate_synthetic_vehicles.py`                      |
| unknown  | 1500    | `generate_synthetic_unknown.py`                       |
| **Total**| **6000**| `merge_windows.py` -> `combined_windows.npz`         |

### Real-human pipeline (the fix this run)

The ICLegend dataset is request-only; we substitute **RadHAR** (https://github.com/nesl/RadHAR), a public 24 GHz mmWave point-cloud dataset from the same lab. RadHAR ships ROS-YAML frames for 5 activities (walk, boxing, jack, jump, squats).

Conversion pipeline (`scripts/convert_iclegend_to_rd03d.py`):
1. Parse each `.txt` file as ROS YAML (15 lines per point, `---` separator, `point_id` resets to 0 at each new frame).
2. Run DBSCAN (`eps=0.5 m`, `min_samples=3`) on the (x,y,z) point cloud.
3. Sort clusters by point-count descending and keep the top 3 (slot 0 = largest = subject, slot 1-2 = reflections/ghosts).
4. Project to RD-03D 2D target format: drop z, compute `d = sqrt(x^2+y^2)`, keep (d, v, x, y).
5. Slide a 16-frame window with stride 4 -> `(16, 3, 4)` tensor.

Result: **32 233 windows** from 142 .txt files (boxing 6 352, jack 6 174, jump 5 889, squats 6 381, walk 7 437).
Down-sampled to 3 000 windows (seed 42) to balance against the synthetic vehicle/unknown sets.

Per-class distance / speed distribution (after cluster reordering):
```
human    target0_d ~ 0.5m, |v|_max ~ 0.2 m/s, persistence ~ 1.0
vehicle  target0_d ~ 4.0m, |v|_max ~ 7.6 m/s, persistence ~ 1.0
unknown  target0_d ~ 4.0m, |v|_max ~10.4 m/s, persistence ~ 0.6
```

---

## Model A: MLP (32 -> 16 -> 3)

| Metric                     | Value     |
|----------------------------|-----------|
| **Test accuracy (float)**  | **100.0%**|
| F1 (human)                 | 1.000     |
| F1 (vehicle)               | 1.000     |
| F1 (unknown)               | 1.000     |
| Macro F1                   | 1.000     |
| **Test accuracy (INT8)**   | **100.0%**|
| **Model size (INT8)**      | **6.6 KB**|
| PC inference latency (INT8)| 0.03 ms   |

Confusion matrix (INT8, n=1200):
```
           human   vehicle   unknown
   human     600         0         0
 vehicle       0       300         0
 unknown       0         0       300
```

### Pass criteria (Layer 1)

- [x] Model size <= 50 KB (**6.6 KB**)
- [x] Overall accuracy >= 80% (**100.0%**)
- [x] F1 (human) >= 0.85 (**1.000**)
- [x] F1 (vehicle) >= 0.80 (**1.000**)
- [x] F1 (unknown) >= 0.70 (**1.000**)

---

## Model B: 1D-CNN (alternative)

| Metric                     | Value     |
|----------------------------|-----------|
| Test accuracy (float)      | 100.0%    |
| F1 (human)                 | 1.000     |
| F1 (vehicle)               | 1.000     |
| F1 (unknown)               | 1.000     |

Saved as `models/model_1dcnn.keras` (54 KB). Run `quantize_export.py` to produce the C array when ready to deploy.

---

## Artifacts produced

| File | Size | Purpose |
|------|------|---------|
| `data/rd03d_dataset.npz`           | 1.9 MB | Real RadHAR -> RD-03D-format window dataset (32 233 windows) |
| `data/real_humans_features_balanced.npz` | ~700 KB | 3 000 windows of real-human features (72-dim) |
| `data/combined_windows.npz`        | 1.2 MB | Real humans + synthetic vehicles/unknown windows for 1D-CNN |
| `data/combined_features.npz`       | 542 KB | Real humans + synthetic vehicles/unknown features for MLP |
| `models/model.keras`               | 63 KB  | Keras MLP (float) |
| `models/model_1dcnn.keras`         | 54 KB  | Keras 1D-CNN (float) |
| `models/model_int8.tflite`         | 6.6 KB | INT8 quantized model -- **ESP32-S3 target** |
| `models/model_data.cc`             | 42 KB  | C array for ESP32-S3 firmware |
| `models/model_metadata.h`          | 2.1 KB | Class labels, normalization params |
| `models/norm_params.npz`           | 1.1 KB | Z-score normalization params |

---

## Notes

1. **The 100% accuracy is partially attributable to the synthetic generators being too clean.** Real RadHAR humans are noisier than the synthetic vehicles/unknown. The model is learning real physical differences (humans near + slow, vehicles farther + faster, unknown intermittent), but expect F1 to drop 5-15 pp on real RD-03D data.

2. **Real-world calibration pass needed** when RD-03D arrives (Week 9).
   Plan: collect 100-200 live samples, fine-tune the trained model.

3. **Next phase**: Phase 3 firmware -- ESP-IDF skeleton with UART parser,
   TFLite Micro inference, FreeRTOS task scheduling, Wi-Fi dashboard.

---

## Calibration on held-out Test split

RadHAR ships a separate `Data/Test/` split (different recording sessions than Train: e.g., walk_5-9 vs walk_1-4). We re-ran the conversion pipeline on it (9 668 windows from 70 .txt files) and evaluated the deployed INT8 model — no re-training, no re-fitting normalization.

| Split                          | n    | Acc    | Logit margin |
|--------------------------------|-----:|-------:|-------------:|
| Real RadHAR Test humans        | 2000 | 100.0% | 0.996        |
| Synthetic vehicles             | 1500 | 100.0% | 0.996        |
| Synthetic unknown              | 1500 | 100.0% | 0.996        |
| **Combined**                   | 5000 | 100.0% | 0.996        |

Logit margin = 0.996 = INT8 saturation at +127/-128. The model is making saturated decisions on every sample, not "barely-confident 100%."

Per-activity accuracy on real Test humans: boxing 100%, jack 100%, jump 100%, squats 100%, walk 100% (all at logit margin 0.996).

### Noise stress test

To bound tolerance to real RD-03D noise, we add Gaussian noise (σ in normalized-feature stdev units) and re-evaluate:

| σ    | Acc    |
|-----:|-------:|
| 0.00 | 100.0% |
| 0.50 | 100.0% |
| 1.00 | 99.1%  |
| 1.50 | 95.8%  |
| 2.00 | 91.3%  |
| 3.00 | 80.5%  |

The model holds >99% accuracy up to σ=1.0 per-feature noise. Degrades to the 80% pass-criteria floor only at σ=3.0. Humans are more robust than vehicles under noise; vehicles are the first to misclassify.

Full report at [CALIBRATION_REPORT.md](phase2/CALIBRATION_REPORT.md). Scripts: `scripts/calibration_v2.py`, `scripts/noise_stress_test.py`.

---

## Reproduce

```bash
# Activate venv
source venv/Scripts/activate        # Git Bash / Linux
# or:  venv\Scripts\activate         # cmd

# Step 1: convert real RadHAR data to RD-03D-format windows
python scripts/convert_iclegend_to_rd03d.py \
    --root data/iclegend_raw/RadHAR/Data/Train \
    --out data/rd03d_dataset.npz \
    --window 16 --stride 4

# Step 2: subsample real humans to balance with synthetic
python -c "
import numpy as np
rng = np.random.default_rng(42)
d = np.load('data/rd03d_dataset.npz')
for key in ['tensors','masks','classes','activities']:
    pass
t, m, c, a = d['tensors'], d['masks'], d['classes'], d['activities']
idx = rng.choice(len(t), size=3000, replace=False)
np.savez_compressed('data/real_humans_windows_balanced.npz',
    tensors=t[idx], masks=m[idx], classes=c[idx], activities=a[idx])
"

# Step 3: merge windows for 1D-CNN
python scripts/merge_windows.py \
    --in data/real_humans_windows_balanced.npz \
         data/synthetic_vehicles.npz data/synthetic_unknown.npz \
    --out data/combined_windows.npz

# Step 4: compute features for MLP
python scripts/compute_features.py --in data/rd03d_dataset.npz \
    --out data/real_humans_features.npz
python scripts/compute_features.py --in data/synthetic_vehicles.npz \
    --out data/synthetic_vehicles_features.npz
python scripts/compute_features.py --in data/synthetic_unknown.npz \
    --out data/synthetic_unknown_features.npz

python -c "
import numpy as np
rng = np.random.default_rng(42)
d = np.load('data/real_humans_features.npz')
X, y = d['X'], d['y']
idx = rng.choice(len(X), size=3000, replace=False)
np.savez_compressed('data/real_humans_features_balanced.npz', X=X[idx], y=y[idx])
"

python scripts/combine_datasets.py \
    --in data/real_humans_features_balanced.npz \
         data/synthetic_vehicles_features.npz data/synthetic_unknown_features.npz \
    --out data/combined_features.npz

# Step 5: train + quantize
python scripts/train_mlp.py    --data data/combined_features.npz --out models/
python scripts/train_1dcnn.py  --data data/combined_windows.npz  --out models/
python scripts/quantize_export.py \
    --keras models/model.keras --norm models/norm_params.npz \
    --data data/combined_features.npz --out models/

# Step 6: verify
python scripts/test_model_offline.py \
    --model models/model_int8.tflite \
    --data  data/combined_features.npz \
    --norm  models/norm_params.npz
```

## Phase 3 — Firmware Skeleton (Weeks 7–8)

Goal: build ESP-IDF firmware skeleton validated end-to-end on host
before the RD-03D sensor arrives in Week 9.

### 2026-09-10: Layer-2 build-up

Feature parity fix (the bug we left off with):

- C `feature_extract.c` was iterating per-target then time for the
  global `d_all`/`v_all` flatten. Python uses `tensor[:,:,0][mask>0.5]`
  which is time-major (C-order). Fixed by adding a `present_pos[k][t]`
  lookup that records where each (t, k) entry lands in `d_series[k]`,
  then iterating time-major for the global flatten.
- Test fixture also had a bug: `out_py` used un-quantized floats while
  C reads int16. Two cases failed on boundary values (|v|=5.001 → 5.00
  after int16 round-trip, changing the `|v|>5.0` count). Test now
  applies int16 quantization to the Python reference first.
- Result: **50/50 trials passed**, max element diff 3e-3 (float32 ε).

Phase 3 deliverables landed:

```
firmware/CMakeLists.txt           ESP-IDF project root
firmware/sdkconfig.defaults       ESP32-S3 + PSRAM + softAP
firmware/partitions.csv           single-app 1.5 MB
firmware/main.c                   4-task FreeRTOS entry
firmware/radar_uart.{h,c}         UART2 driver stub (host harness
                                  calls radar_parse_frame directly)
firmware/tracker.{h,c}            alpha-beta tracker
firmware/inference.{h,c}          TFLite Micro wrapper + int_softmax
firmware/dashboard_http.{h,c}     JSON serializers
firmware/main/dashboard/index.html
firmware/main/dashboard/app.js    canvas radar plot
firmware/main/dashboard/style.css dark theme
tests/loopback_test.py            end-to-end host harness
firmware/README.md                build + verify instructions
scripts/build_firmware.bat        extended to compile all 5 modules
firmware/radar_parse.def          new exports (tracker, inference, json)
firmware/test_tracker.c           host-side unit test
```

Verification commands:

```bash
# Build the host-side DLL (Windows + MSVC build tools)
cmd /c scripts\build_firmware.bat

# Round-trip + CRC reference vector
python scripts/protocol_encode.py --self-test

# 50 random windows, max element diff within float32 epsilon
python tests/cross_check_features.py

# Full pipeline: simulator -> C parser -> C features -> Python
# inference (when tflite_runtime available) -> Python tracker
python tests/loopback_test.py --frames 5000
```

Latest run:

```
protocol_encode.py --self-test:    All protocol self-tests passed.
cross_check_features.py:           50/50 trials, max diff 3.019e-03.
loopback_test.py --frames 1000:    1000 parsed, 62 windows, 0 failures,
                                   PASS.
```

Pending before Week 9:

- Install ESP-IDF v5.2+ and `espressif/esp-tflite-micro` component to
  enable `idf.py build` of `main.c`. The skeleton is complete; only
  the TFLite Micro interpreter wiring in `inference.c`/`main.c` is
  needed.
- Connect real RD-03D on UART2 and rerun the same harness against the
  physical sensor.
- Soak test: 24-hour replay of recorded dataset against ESP32-S3.

### 2026-09-10 (continued): real inference + local dashboard

The Phase 3 loopback harness was passing parser/feature/tracker tests,
but inference was using a stub. Replaced with a real NumPy MLP forward
pass (read from `model.keras` via `h5py`) so the harness actually
classifies firmware-extracted features.

Verification on `combined_windows.npz` (3000 human / 1500 vehicle /
1500 unknown, shuffled):

```
classification acc:   498/500 = 99.6%
Per-class confusion:
  human:   100.0%        vehicle: 100.0%      unknown: 98.1%
```

`tracker.c` had a publish-gating bug: `tk->age_s` was reset to 0 on
every matched frame, so the cooldown gate never opened. Fixed by
adding a `hit_count` field that publishes after 3 consecutive hits
(at 20 Hz, that's ~150 ms; sufficient to suppress one-frame flicker).

Built `tests/dashboard_server.py` — a local HTTP server on
`127.0.0.1:8080` that serves the firmware dashboard HTML/CSS/JS
verbatim and streams live `/api/state` JSON from the loopback
pipeline. Open `http://127.0.0.1:8080/` to see the same UI the
ESP32-S3 will serve over its softAP, populated with real
parser -> features -> inference -> tracker output.

```bash
# Start the dashboard (browser auto-opens on Windows)
python tests/dashboard_server.py --frames 8000 --shuffle

# Or run headless (curl the JSON)
curl http://127.0.0.1:8080/api/state
```

The radar canvas now uses 800x800 px, range rings to 64 m, class
colors green/blue/gray for human/vehicle/unknown. Tracks show class,
confidence, position, speed, age, and hit count. Tracks with
`hit_count < 3` are dimmed as pending.

Dashboard screenshot:
- Header: Uptime 110s, FPS 19.9, Infer 0.1 ms, Frames 2208
- 16 tracks visible: humans (green), vehicles (blue), unknowns (gray)
- Track #138 human (-21.58, 3.06) m, v=1.40 m/s, age=16.8, hits=2

