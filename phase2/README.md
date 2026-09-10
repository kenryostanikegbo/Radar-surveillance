# Phase 2: Dataset Acquisition → Trained Model

**Goal**: Build a TFLite INT8 model that classifies radar targets as human/vehicle/unknown, trained on a 24 GHz FMCW dataset that matches the RD-03D's output format.

**Timeline**: Weeks 4–6 (3 weeks)

---

## Week 4 — Dataset Acquisition & Conversion

- [ ] Download ICLegend / MDPI 24 GHz Fall Detection & Activity Dataset
- [ ] Inspect raw file structure (point cloud format, label files)
- [ ] Run `scripts/convert_iclegend_to_rd03d.py` → produces `data/rd03d_dataset.npz`
- [ ] Verify tensor shape `(N, 16, 3, 4)` and class balance
- [ ] Visualise 3 sample windows per class for sanity check

## Week 5 — Synthetic Augmentation & EDA

- [ ] Run `scripts/generate_synthetic_vehicles.py` to add vehicle samples
- [ ] Run `scripts/generate_synthetic_unknown.py` to add clutter/unknown samples
- [ ] Aim for ≥1500 windows per class (human, vehicle, unknown)
- [ ] Complete `notebooks/EDA.ipynb`:
  - Range distribution per class
  - Speed distribution per class
  - Class separability plots
  - Missing-target rate per class

## Week 6 — Model Training & Export

- [ ] Run `scripts/train_mlp.py` (sanity check baseline, ≥80% F1)
- [ ] Run `scripts/train_1dcnn.py` (1D-CNN on time windows)
- [ ] Run `scripts/quantize_export.py` → produces `model_int8.tflite` (≤50 KB)
- [ ] Generate C array: `xxd -i model_int8.tflite > model_data.cc`
- [ ] Run `scripts/test_model_offline.py` for verification

---

## Deliverables at End of Week 6

1. `data/rd03d_dataset.npz` — converted ICLegend + synthetic data
2. `models/model_int8.tflite` — INT8 quantized model ≤50 KB
3. `models/model_data.cc` — C array header for ESP32-S3
4. `models/model_metadata.h` — class labels, feature scaling parameters
5. `notebooks/EDA.ipynb` — completed exploratory data analysis
6. `scripts/` — all training and conversion scripts
7. `README.md` — instructions for reproducing the model

---

## Pass Criteria

- [ ] Tensor dataset shape `(≥5000, 16, 3, 4)` with class balance within 30%
- [ ] INT8 model file size ≤ 50 KB
- [ ] F1-score (human): ≥ 0.85
- [ ] F1-score (vehicle): ≥ 0.80
- [ ] F1-score (unknown): ≥ 0.70
- [ ] PC inference latency < 5 ms (sanity)

When all criteria met, hand off to Phase 3 (firmware deployment).

---

## Layer 1 Verification (per TESTING_STRATEGY.md)

After Week 6, run `scripts/test_model_offline.py` to confirm:
- Model accuracy matches training
- INT8 quantization did not break anything
- File size fits ESP32-S3 flash budget

---

## Notes

- RD-03D has not arrived. All work uses ICLegend dataset projected to RD-03D format.
- When RD-03D arrives (Week 9), plan a 1–2 day recalibration pass per TESTING_STRATEGY.md.
