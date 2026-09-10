# Exploratory Data Analysis Checklist

Run through this checklist once you have `data/combined_windows.npz`.

Replace the cells with actual code as you explore.

---

## 1. Load and inspect

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.load("data/combined_windows.npz")
tensors = data["tensors"]   # (N, 16, 3, 4)
masks   = data["masks"]     # (N, 16, 3)
classes = data["classes"]

print(f"Tensors: {tensors.shape}")
print(f"Masks:   {masks.shape}")
print(f"Classes: {dict(zip(*np.unique(classes, return_counts=True)))}")
```

## 2. Class balance

Are all classes roughly balanced? If not, consider:
- Oversampling minority class
- Class weights during training
- Generating more synthetic samples

## 3. Range (d) distribution per class

```python
for cls in ["human", "vehicle", "unknown"]:
    mask = classes == cls
    d_values = tensors[mask, :, :, 0][masks[mask] > 0].flatten()
    plt.hist(d_values, bins=50, alpha=0.5, label=cls)
plt.xlabel("Distance (m)")
plt.ylabel("Count")
plt.legend()
plt.title("Range distribution per class")
```

**Expected**:
- Human: 0.5–7 m (dataset's subjects walking)
- Vehicle: 1–7 m (synthetic generator range)
- Unknown: 0.5–7.5 m (broad)

## 4. Speed distribution per class

```python
for cls in ["human", "vehicle", "unknown"]:
    mask = classes == cls
    v_values = tensors[mask, :, :, 1][masks[mask] > 0].flatten()
    plt.hist(v_values, bins=100, alpha=0.5, label=cls, range=(-15, 15))
plt.xlabel("Radial velocity (m/s)")
plt.ylabel("Count")
plt.legend()
plt.title("Speed distribution per class")
```

**Expected**:
- Human: 0–2 m/s (walking)
- Vehicle: -15 to 15 m/s (driving)
- Unknown: broad, possibly noisy

## 5. Detection persistence per class

```python
for cls in ["human", "vehicle", "unknown"]:
    mask = classes == cls
    persistence = masks[mask, :, 0].sum(axis=1) / 16  # primary target persistence
    plt.hist(persistence, bins=20, alpha=0.5, label=cls)
plt.xlabel("Primary target persistence (fraction of frames)")
plt.ylabel("Count")
plt.legend()
plt.title("How often the primary target is visible per window")
```

**Expected**:
- Human: high persistence (consistent detection)
- Vehicle: very high persistence (strong RCS)
- Unknown: variable (intermittent clutter)

## 6. Visualize a sample window per class

```python
for cls in ["human", "vehicle", "unknown"]:
    idx = np.where(classes == cls)[0][0]
    window = tensors[idx]   # (16, 3, 4)
    mask_w = masks[idx]     # (16, 3)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for k in range(3):
        d = window[:, k, 0]
        valid = mask_w[:, k] > 0.5
        axes[0].plot(d[valid], '.', label=f"target {k}")
    axes[0].set_title(f"Range over time — {cls}")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Distance (m)")
    axes[0].legend()

    for k in range(3):
        v = window[:, k, 1]
        valid = mask_w[:, k] > 0.5
        axes[1].plot(v[valid], '.', label=f"target {k}")
    axes[1].set_title(f"Speed over time — {cls}")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Speed (m/s)")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
```

## 7. (Advanced) t-SNE on feature vectors

After running `compute_features.py`, visualize 2D embedding:

```python
from sklearn.manifold import TSNE
import os, numpy as np

feat = np.load("data/combined_features.npz")
tsne = TSNE(n_components=2, random_state=42)
X_2d = tsne.fit_transform(feat["X"])

for cls in ["human", "vehicle", "unknown"]:
    mask = feat["y"] == cls
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1], alpha=0.5, label=cls)
plt.legend()
plt.title("t-SNE of features by class")
```

**Look for**: visible clusters / good separation between classes.

---

## Pass criteria for EDA

- [ ] Class balance within 30% (no class < 70% of the largest)
- [ ] At least one feature clearly differs between classes
- [ ] No obvious outliers / corrupt frames
- [ ] t-SNE shows some class separation
