"""
Concatenate multiple window-format npz files (same schema) into one.

Usage:
    python merge_windows.py \
        --in data/rd03d_dataset.npz data/synthetic_vehicles.npz data/synthetic_unknown.npz \
        --out data/combined_windows.npz
"""

import argparse
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    tensors, masks, classes, activities = [], [], [], []

    for path in args.inputs:
        d = np.load(path)
        tensors.append(d["tensors"])
        masks.append(d["masks"])
        classes.append(d["classes"])
        activities.append(d["activities"])
        print(f"{path}: tensors {d['tensors'].shape}, "
              f"class dist {dict(zip(*np.unique(d['classes'], return_counts=True)))}")

    out = {
        "tensors":   np.concatenate(tensors, axis=0),
        "masks":     np.concatenate(masks, axis=0),
        "classes":   np.concatenate(classes, axis=0),
        "activities":np.concatenate(activities, axis=0),
    }

    np.savez_compressed(args.out, **out)
    print(f"\nCombined: {out['tensors'].shape} -> {args.out}")
    print(f"Class dist: {dict(zip(*np.unique(out['classes'], return_counts=True)))}")


if __name__ == "__main__":
    main()
