"""
Generate synthetic radar target reports for vehicles.

RD-03D signature characteristics for vehicles (vs humans):
  - Larger RCS -> stronger, more stable returns (no missing frames)
  - Higher speeds typical (5–15 m/s in a parking/road scene)
  - Smoother trajectory (vehicles do not swing arms/legs)
  - Larger spatial extent (cluster spans wider x,y range)
  - Less variance in radial velocity over short windows

We synthesise a target sequence that walks from a start position to
an end position with a chosen speed, and emits the same kind of
multi-target report frames the RD-03D would emit.

Output: npz with same schema as rd03d_dataset.npz so it can be
        concatenated before training.

Usage:
    python generate_synthetic_vehicles.py \
        --n-samples 2000 \
        --out data/synthetic_vehicles.npz
"""

import argparse
import numpy as np
from pathlib import Path


def simulate_vehicle_trajectory(
    n_frames,
    dt=0.1,                 # 10 Hz like RD-03D
    speed_range=(2.0, 15.0),# m/s
    range_clip=(0.5, 7.5),  # RD-03D practical range
    angle_range=(-50, 50),  # degrees from boresight
    missing_prob=0.0,       # vehicles rarely drop frames
):
    """
    Generate one (window, mask) tuple representing a single vehicle
    moving through the radar's field of view.
    """
    window_size = 16
    max_targets = 3
    tensor = np.zeros((window_size, max_targets, 4), dtype=np.float32)
    mask   = np.zeros((window_size, max_targets), dtype=np.float32)

    speed = np.random.uniform(*speed_range)
    angle_deg = np.random.uniform(*angle_range)
    angle_rad = np.deg2rad(angle_deg)

    # Direction: towards or away from radar
    direction = np.random.choice([-1.0, 1.0])
    # If angle is small, the radial component is mostly the speed itself
    v_radial = speed * np.cos(angle_rad) * direction

    # Starting distance
    d0 = np.random.uniform(range_clip[0] + 1, range_clip[1] - 1)

    for t in range(window_size):
        d = d0 + v_radial * (t * dt)

        # Stop when we leave the range clip — restart at other edge
        if d < range_clip[0] or d > range_clip[1]:
            d0 = np.random.uniform(range_clip[0] + 1, range_clip[1] - 1)
            d = d0
            v_radial = -v_radial

        # Convert (d, angle) -> (x, y)
        x = d * np.sin(angle_rad)
        y = d * np.cos(angle_rad)

        # Add small jitter (vehicle engine vibration, path wobble)
        x += np.random.normal(0, 0.05)
        y += np.random.normal(0, 0.05)
        d_noisy = np.sqrt(x**2 + y**2)

        # Sometimes emit a stationary false detection (clutter)
        # to make the synthetic data more realistic
        tensor[t, 0, 0] = d_noisy
        tensor[t, 0, 1] = v_radial + np.random.normal(0, 0.05)
        tensor[t, 0, 2] = x
        tensor[t, 0, 3] = y
        mask[t, 0] = 1.0

        # 5% chance of a secondary reflection (e.g. trailer, side mirror)
        if np.random.random() < 0.05 and max_targets > 1:
            offset = np.random.uniform(0.3, 0.8)
            tensor[t, 1, 0] = max(0.1, d_noisy + offset)
            tensor[t, 1, 1] = v_radial + np.random.normal(0, 0.1)
            tensor[t, 1, 2] = x + np.random.uniform(-0.3, 0.3)
            tensor[t, 1, 3] = y + np.random.uniform(-0.3, 0.3)
            mask[t, 1] = 1.0

    return tensor, mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--out", default="data/synthetic_vehicles.npz")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)

    tensors, masks = [], []
    classes = []
    activities = []

    for i in range(args.n_samples):
        tensor, mask = simulate_vehicle_trajectory(n_frames=16)
        tensors.append(tensor)
        masks.append(mask)
        classes.append("vehicle")
        activities.append("synthetic_vehicle")

    tensors = np.stack(tensors)
    masks = np.stack(masks)
    classes = np.array(classes)
    activities = np.array(activities)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out,
                        tensors=tensors, masks=masks,
                        classes=classes, activities=activities)

    print(f"Generated {args.n_samples} vehicle windows -> {args.out}")
    print(f"Tensor shape: {tensors.shape}")
    print(f"Class distribution: {dict(zip(*np.unique(classes, return_counts=True)))}")


if __name__ == "__main__":
    main()
