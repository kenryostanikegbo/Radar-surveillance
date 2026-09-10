"""
Merge multiple feature npz files into one combined dataset.

Usage:
    python combine_datasets.py \
        --in data/features.npz data/synthetic_vehicles_features.npz data/synthetic_unknown_features.npz \
        --out data/combined_features.npz
"""

import argparse
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    X_all, y_all = [], []
    for path in args.inputs:
        d = np.load(path)
        X_all.append(d["X"])
        y_all.append(d["y"])
        print(f"{path}: {d['X'].shape} -> class dist {dict(zip(*np.unique(d['y'], return_counts=True)))}")

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    np.savez_compressed(args.out, X=X, y=y)
    print(f"\nCombined: {X.shape} -> {args.out}")
    print(f"Final class dist: {dict(zip(*np.unique(y, return_counts=True)))}")


if __name__ == "__main__":
    main()
