"""
Generate synthetic 'unknown' / clutter target reports.

These represent noisy detections that are NOT human and NOT vehicle:
  - Wind-blown vegetation (random Doppler, intermittent)
  - Insects close to the radar (high RCS variation, very short range)
  - Reflections off fences / walls / buildings (stationary or slow)
  - Random noise bursts
  - Multi-path ghosts

Key signature characteristics:
  - Highly variable detection presence (frame-to-frame dropout)
  - Erratic, non-physical trajectories
  - Speed values inconsistent (jumping around)
  - Sometimes very large, sometimes very small

Output: npz with same schema as rd03d_dataset.npz.

Usage:
    python generate_synthetic_unknown.py \
        --n-samples 2000 \
        --out data/synthetic_unknown.npz
"""

import argparse
import numpy as np
from pathlib import Path


def simulate_clutter_burst(n_frames=16, max_targets=3):
    """One window of clutter/unknown returns."""
    tensor = np.zeros((n_frames, max_targets, 4), dtype=np.float32)
    mask = np.zeros((n_frames, max_targets), dtype=np.float32)

    # Decide how many 'targets' this clutter scene has (1-3)
    n_active = np.random.choice([1, 1, 2, 2, 3])  # skewed toward 1-2

    for t in range(n_frames):
        for k in range(n_active):
            # 60% chance the target is present in this frame
            if np.random.random() < 0.4:
                continue

            # Pick a random behaviour mode
            mode = np.random.choice(["stationary", "erratic", "fast_noise"])

            if mode == "stationary":
                # Wall / fence reflection: stays put
                if t == 0 or mask[t-1, k] == 0:
                    d = np.random.uniform(1.0, 7.0)
                    angle = np.deg2rad(np.random.uniform(-50, 50))
                    x = d * np.sin(angle)
                    y = d * np.cos(angle)
                # else: keep previous x,y
                speed = np.random.normal(0, 0.1)  # tiny jitter only

            elif mode == "erratic":
                # Vegetation in wind: random walk
                if t == 0 or mask[t-1, k] == 0:
                    d = np.random.uniform(1.0, 7.0)
                    angle = np.deg2rad(np.random.uniform(-50, 50))
                    x = d * np.sin(angle)
                    y = d * np.cos(angle)
                else:
                    x += np.random.normal(0, 0.4)
                    y += np.random.normal(0, 0.4)
                speed = np.random.normal(0, 1.0)

            else:  # fast_noise
                # Random noise burst
                d = np.random.uniform(0.5, 7.5)
                angle = np.deg2rad(np.random.uniform(-50, 50))
                x = d * np.sin(angle)
                y = d * np.cos(angle)
                speed = np.random.uniform(-10, 10)

            # Occasionally emit an impossible value (multi-path ghost)
            if np.random.random() < 0.1:
                speed *= np.random.choice([-3, -2, 5])

            d = np.sqrt(x**2 + y**2)
            tensor[t, k, 0] = d
            tensor[t, k, 1] = speed
            tensor[t, k, 2] = x
            tensor[t, k, 3] = y
            mask[t, k] = 1.0

    return tensor, mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--out", default="data/synthetic_unknown.npz")
    p.add_argument("--seed", type=int, default=43)
    args = p.parse_args()

    np.random.seed(args.seed)

    tensors, masks = [], []
    classes, activities = [], []

    for i in range(args.n_samples):
        tensor, mask = simulate_clutter_burst()
        tensors.append(tensor)
        masks.append(mask)
        classes.append("unknown")
        activities.append("synthetic_clutter")

    tensors = np.stack(tensors)
    masks = np.stack(masks)
    classes = np.array(classes)
    activities = np.array(activities)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out,
                        tensors=tensors, masks=masks,
                        classes=classes, activities=activities)

    print(f"Generated {args.n_samples} unknown/clutter windows -> {args.out}")
    print(f"Tensor shape: {tensors.shape}")
    print(f"Class distribution: {dict(zip(*np.unique(classes, return_counts=True)))}")


if __name__ == "__main__":
    main()
