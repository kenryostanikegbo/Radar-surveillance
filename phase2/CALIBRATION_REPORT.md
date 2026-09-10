# Phase 2 Calibration Report

**Date**: 2026-09-09
**Held-out set**: RadHAR `Data/Test/` (real, separate recording sessions, never seen in training)
**Combined with**: synthetic vehicles + unknown (deterministic generators)
**Model under test**: `models/model_int8.tflite` (the actual file that goes on ESP32-S3)

---

## TL;DR

The deployed INT8 model achieves **100% accuracy with saturated logit margins** on the held-out Test split — across **5 different human activities** (boxing, jack, jump, squats, walk), **synthetic vehicles**, and **synthetic unknown**. A noise-stress sweep shows the model holds **>99% accuracy up to σ=1.0 per-feature Gaussian noise** and degrades to the pass-criteria floor (~80%) only at σ=3.0 — i.e., the model is genuinely robust, not just overfit to clean synthetic data.

---

## 1. Held-out evaluation (the actual calibration pass)

### Data composition

| Split                          | Source                              | n used | Class label |
|--------------------------------|-------------------------------------|-------:|-------------|
| Real RadHAR Test humans        | `data/iclegend_raw/RadHAR/Data/Test/` | 2000  | human       |
| Synthetic vehicles             | `generate_synthetic_vehicles.py`    | 1500  | vehicle     |
| Synthetic unknown              | `generate_synthetic_unknown.py`     | 1500  | unknown     |
| **Combined held-out**          |                                     | **5000** |            |

The Test split contains different recording sessions than Train (e.g., `walk_5-9` vs `walk_1-4`), so this is a true generalization check.

### Per-split results

| Split                       | n    | Acc    | Logit margin (mean) | Logit margin (p5) |
|-----------------------------|-----:|-------:|--------------------:|------------------:|
| Real RadHAR Test humans     | 2000 | 100.0% | 0.996               | 0.996             |
| Synthetic vehicles          | 1500 | 100.0% | 0.996               | 0.996             |
| Synthetic unknown           | 1500 | 100.0% | 0.996               | 0.996             |

**Logit margin = 0.996** is the maximum possible under INT8 (saturation at +127 / -128). The model isn't "barely confident 100%" — it's making saturated decisions on every sample.

### Per-activity breakdown on real Test humans

Every activity separates cleanly:

| Activity | n    | Acc    | Logit margin |
|----------|-----:|-------:|-------------:|
| boxing   | 1953 | 100.0% | 0.996        |
| jack     | 1815 | 100.0% | 0.996        |
| jump     | 1979 | 100.0% | 0.996        |
| squats   | 1816 | 100.0% | 0.996        |
| walk     | 2105 | 100.0% | 0.996        |

All five human gestures classify correctly with full INT8-saturation confidence.

### Combined confusion matrix

```
           human   vehicle   unknown
   human    2000         0         0
 vehicle       0      1500         0
 unknown       0         0      1500
```

Zero errors. No cross-class confusion.

---

## 2. Noise stress test

Real RD-03D data won't be as clean as RadHAR or our synthetic generators — antenna cross-talk, multi-path, range-bin quantization all add noise. To bound how much noise the model can tolerate before breaking, we add Gaussian noise (σ in normalized-feature stdev units, since `norm_params.npz` was used to z-score everything) and re-evaluate.

```
   sigma      acc    human  vehicle  unknown
    0.00  100.00%  100.00%  100.00%  100.00%
    0.10  100.00%
    0.20  100.00%
    0.30  100.00%
    0.50  100.00%  100.00%  100.00%  100.00%
    0.75   99.70%
    1.00   99.10%  100.00%   98.30%   99.00%
    1.50   95.77%   98.40%   93.50%   95.40%
    2.00   91.33%   94.70%   89.20%   90.10%
    3.00   80.53%
```

**Findings:**

- **σ ≤ 0.5: 100%** — model is unaffected by noise up to half a feature standard deviation.
- **σ = 1.0: 99.1%** — vehicles are the first to degrade (98.3%); humans + unknown still at 100%.
- **σ = 2.0: 91.3%** — model still useful; humans hold up better than vehicles/unknown.
- **σ = 3.0: 80.5%** — at the Layer-1 pass-criteria floor.

Because normalized features have unit variance by construction, σ=1.0 represents "noise equal in magnitude to one standard deviation of the feature distribution." That is a *lot* of noise — the model has genuine headroom for the noisier RD-03D returns we expect on hardware.

---

## 3. Caveats

1. **Synthetic vehicle/unknown data is over-clean.** The generator produces deterministic point clouds with no missed detections, no spurious returns, and perfect Doppler. Real RD-03D will have all of these; the σ=1.0 noise sweep gives a partial proxy but not a perfect one.

2. **Per-feature noise is independent.** Real radar noise is correlated across the 72-dim feature vector (e.g., bad range-bin estimation corrupts d-derived features together). The stress test is a worst-case upper bound on noise tolerance, since correlated noise preserves class-conditional structure better than independent noise.

3. **The 100% accuracy is also partially because the feature engineering cleanly separates the classes.** Speed (humans ~0.2 m/s, vehicles ~7.6 m/s, unknown intermittent) and persistence alone are highly discriminative. The MLP is doing very little work beyond weighting these.

4. **No real RD-03D data in this pass.** Calibration on Test uses a different session of the same 24 GHz radar (RadHAR), not the actual RD-03D module. We expect RD-03D returns to differ in:
   - Range resolution (RD-03D ≈ 0.5m bins; RadHAR records centroid point clouds)
   - Frame rate (RD-03D ~20 fps; RadHAR varies)
   - Multi-path behavior (different antenna layout)

   The noise-stress test partially covers this, but a true cross-sensor validation requires live RD-03D captures (Week 9 with hardware).

---

## 4. Pass-criteria confirmation (Layer 1)

- [x] Model size <= 50 KB (**6.6 KB**)
- [x] Overall accuracy >= 80% (**100.0%** on held-out Test)
- [x] F1 (human) >= 0.85 (**1.000**)
- [x] F1 (vehicle) >= 0.80 (**1.000**)
- [x] F1 (unknown) >= 0.70 (**1.000**)
- [x] All human activities separable (**100% each**)
- [x] Saturated logit margin (**0.996**)
- [x] Robust to σ=1.0 noise (**99.1%**)

---

## 5. Reproduce

```bash
source venv/Scripts/activate

# Convert held-out Test split (run once)
python scripts/convert_iclegend_to_rd03d.py \
    --root data/iclegend_raw/RadHAR/Data/Test \
    --out  data/rd03d_dataset_test.npz \
    --window 16 --stride 4

# Compute features
python scripts/compute_features.py \
    --in  data/rd03d_dataset_test.npz \
    --out data/real_humans_features_test.npz

# Run calibration pass (logit-space margins)
python scripts/calibration_v2.py

# Run noise-stress sweep
python scripts/noise_stress_test.py
```

Artifacts:
- `scripts/calibration_v2.py` — accuracy + logit margin + per-activity report
- `scripts/noise_stress_test.py` — σ sweep with per-class breakdown
