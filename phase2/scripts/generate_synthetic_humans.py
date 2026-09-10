"""
Generate synthetic 'human' radar target reports.

Key signature characteristics:
  - Speed 0-2 m/s (typical walking)
  - Speed variance HIGH (limb motion causes micro-Doppler oscillation)
  - Speed sign-change rate moderate (limbs swing forward/back)
  - Persistence HIGH (humans are good RCS reflectors)
  - Trajectory smoother than clutter but wobblier than vehicles

This is a temporary stand-in until the ICLegend dataset arrives.
Replace with `convert_iclegend_to_rd03d.py` when ICLegend is in hand.

Usage:
    python generate_synthetic_humans.py --n-samples 2000 --out data/synthetic_humans.npz
"""

import argparse
import numpy as np
from pathlib import Path


def simulate_human_trajectory(
    n_frames=16,
    dt=0.1,                  # 10 Hz like RD-03D
    speed_range=(0.3, 1.8),  # m/s walking
    range_clip=(0.5, 7.5),
    angle_range=(-50, 50),
):
    """
    Simulate one human walker through the radar FOV.
    Adds gait-induced speed wobble to mimic micro-Doppler signatures.
    """
    window_size = n_frames
    max_targets = 3
    tensor = np.zeros((window_size, max_targets, 4), dtype=np.float32)
    mask   = np.zeros((window_size, max_targets), dtype=np.float32)

    speed = np.random.uniform(*speed_range)
    angle_deg = np.random.uniform(*angle_range)
    angle_rad = np.deg2rad(angle_deg)
    direction = np.random.choice([-1.0, 1.0])
    base_radial = speed * np.cos(angle_rad) * direction

    d0 = np.random.uniform(range_clip[0] + 1, range_clip[1] - 1)

    # Each human has a slightly different gait frequency (1.5-2.5 Hz)
    gait_hz = np.random.uniform(1.4, 2.6)
    gait_phase = np.random.uniform(0, 2 * np.pi)

    for t in range(window_size):
        d = d0 + base_radial * (t * dt)

        if d < range_clip[0] or d > range_clip[1]:
            d0 = np.random.uniform(range_clip[0] + 1, range_clip[1] - 1)
            d = d0
            base_radial = -base_radial

        x = d * np.sin(angle_rad)
        y = d * np.cos(angle_rad)

        # Gait modulation: small periodic wobble in range + speed
        wobble = 0.05 * np.sin(2 * np.pi * gait_hz * t * dt + gait_phase)
        x += wobble * 0.5
        y += wobble * 0.5
        d_noisy = np.sqrt(x**2 + y**2)

        # Speed wobble mimics limb Doppler oscillation
        radial_speed = base_radial + 0.3 * np.sin(2 * np.pi * gait_hz * t * dt + gait_phase + np.pi/2)

        tensor[t, 0, 0] = d_noisy
        tensor[t, 0, 1] = radial_speed
        tensor[t, 0, 2] = x
        tensor[t, 0, 3] = y
        mask[t, 0] = 1.0

    return tensor, mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--out", default="data/synthetic_humans.npz")
    p.add_argument("--seed", type=int, default=44)
    args = p.parse_args()

    np.random.seed(args.seed)

    tensors, masks = [], []
    classes = []
    activities = []

    for i in range(args.n_samples):
        tensor, mask = simulate_human_trajectory()
        tensors.append(tensor)
        masks.append(mask)
        classes.append("human")
        activities.append("synthetic_human")

    tensors = np.stack(tensors)
    masks = np.stack(masks)
    classes = np.array(classes)
    activities = np.array(activities)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out,
                        tensors=tensors, masks=masks,
                        classes=classes, activities=activities)

    print(f"Generated {args.n_samples} human windows -> {args.out}")
    print(f"Tensor shape: {tensors.shape}")
    print(f"Class distribution: {dict(zip(*np.unique(classes, return_counts=True)))}")


if __name__ == "__main__":
    main()
