"""
Convert point-cloud radar dataset (RadHAR format) into RD-03D-equivalent frames.

Input format: ROS YAML, one point per 15-line block:
    header: ...
      seq: ...
      stamp: ...
        secs: ...
        nsecs: ...
      frame_id: "ti_mmwave"
    point_id: 0
    x: 0.0117
    y: -0.039
    z: -0.0078
    range: 0.04
    velocity: 0.0
    doppler_bin: 8
    bearing: -73.3
    intensity: 30.7
    ---
    header: ...
    point_id: 1
    ...

Frame grouping: point_id resets to 0 when a new frame starts.

Usage:
    python convert_iclegend_to_rd03d.py \
        --root data/iclegend_raw/RadHAR/Data/Train \
        --out data/rd03d_dataset.npz \
        --window 16 --stride 4
"""

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN


# ---- Regex to extract fields from each point's YAML block ----
RE_POINT_ID = re.compile(r"point_id:\s*(\d+)")
RE_X  = re.compile(r"^x:\s*([-+]?\d*\.?\d+)", re.MULTILINE)
RE_Y  = re.compile(r"^y:\s*([-+]?\d*\.?\d+)", re.MULTILINE)
RE_Z  = re.compile(r"^z:\s*([-+]?\d*\.?\d+)", re.MULTILINE)
RE_V  = re.compile(r"^velocity:\s*([-+]?\d*\.?\d+)", re.MULTILINE)


def parse_pointcloud_file(txt_path):
    """
    Parse a RadHAR .txt file into a list of (N, 4) point-cloud frames.

    Returns:
        frames: list of np.ndarray, each shape (N, 4) where columns are [x, y, z, v]
    """
    with open(txt_path, "r", errors="ignore") as f:
        content = f.read()

    # Split on the YAML record separator ---
    records = content.split("---")

    frames = []
    current_pts = []

    for rec in records:
        pid_m = RE_POINT_ID.search(rec)
        if pid_m is None:
            continue
        pid = int(pid_m.group(1))

        # If point_id reset to 0 and we already have points, this is a new frame
        if pid == 0 and current_pts:
            if len(current_pts) >= 2:
                frames.append(np.array(current_pts, dtype=np.float32))
            current_pts = []

        # Extract x, y, z, velocity from this record
        x_m = RE_X.search(rec)
        y_m = RE_Y.search(rec)
        z_m = RE_Z.search(rec)
        v_m = RE_V.search(rec)

        if x_m and y_m and z_m and v_m:
            try:
                x = float(x_m.group(1))
                y = float(y_m.group(1))
                z = float(z_m.group(1))
                v = float(v_m.group(1))
                current_pts.append([x, y, z, v])
            except ValueError:
                continue

    # Don't forget the last frame
    if len(current_pts) >= 2:
        frames.append(np.array(current_pts, dtype=np.float32))

    return frames


# ---- Step 1: cluster raw point cloud into <= max_targets centroids ----

def cluster_to_centroids(points_xyz_v, eps=0.5, min_samples=3, max_targets=3):
    """
    Replace raw point cloud with cluster centroids.
    Closest analogue to RD-03D's on-sensor clustering (which caps reported targets).

    points_xyz_v : (N, 4) array -- x, y, z, v_r per detected point
    Returns      : (M, 4) array where M <= max_targets, sorted by cluster size desc.
                   Each row: [cx, cy, cz, mean_v_r]
    """
    if len(points_xyz_v) == 0:
        return np.empty((0, 4))

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_xyz_v[:, :3])
    labels = clustering.labels_

    centroids = []
    label_ids = []
    for lbl in set(labels):
        if lbl == -1:
            continue  # noise
        cluster_pts = points_xyz_v[labels == lbl]
        cx, cy, cz = cluster_pts[:, :3].mean(axis=0)
        cv = cluster_pts[:, 3].mean()
        centroids.append([cx, cy, cz, cv])
        label_ids.append(lbl)

    if len(centroids) == 0:
        return np.empty((0, 4))

    # Sort by cluster size desc so slot 0 = largest cluster = most reliable target.
    counts = np.array([int(np.sum(labels == lbl)) for lbl in label_ids])
    order = np.argsort(counts)[::-1]
    centroids = np.array(centroids)[order]

    if len(centroids) > max_targets:
        centroids = centroids[:max_targets]

    return centroids


# ---- Step 2: project 3D -> RD-03D-format 2D target ----

def project_to_rd03d(points_xyz_v):
    """
    Drop z-axis. Compute d = sqrt(x^2 + y^2).

    Returns list of dicts: {x_m, y_m, d_m, speed_mps, elev_m}
    """
    out = []
    for x, y, z, v in points_xyz_v:
        d = float(np.sqrt(x**2 + y**2))
        out.append({
            "x_m":      float(x),
            "y_m":      float(y),
            "d_m":      d,
            "speed_mps": float(v),
            "elev_m":   float(z),
        })
    return out


# ---- Step 3: walk the dataset ----

def load_dataset(root):
    """
    Walk the dataset directory recursively and parse every .txt file.
    Returns:
        frames : list of (N, 4) numpy arrays (one per radar frame)
        labels : list of activity strings (same length as frames)
    """
    frames, labels = [], []

    txt_files = sorted(glob.glob(os.path.join(root, "**", "*.txt"), recursive=True))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found under {root}")

    print(f"Found {len(txt_files)} .txt files in {root}")

    for i, txt_path in enumerate(txt_files):
        # Activity name = parent folder name (e.g. "walk", "boxing", "jump")
        activity = Path(txt_path).parent.name

        try:
            file_frames = parse_pointcloud_file(txt_path)
        except Exception as e:
            print(f"  [{i+1}/{len(txt_files)}] SKIP {txt_path}: {e}")
            continue

        for frame in file_frames:
            frames.append(frame)
            labels.append(activity)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(txt_files)}] {activity}: "
                  f"{len(file_frames)} frames from {Path(txt_path).name}")

    print(f"\nTotal frames loaded: {len(frames)}")
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Activity distribution: {dict(zip(unique, counts))}")
    return frames, labels


# ---- Step 4: map fine-grained activities -> 3 coarse classes ----

# RadHAR activities -> our 3 coarse classes
ACTIVITY_TO_CLASS = {
    # Human activities
    "walk":    "human",
    "boxing":  "human",
    "jack":    "human",
    "jump":    "human",
    "squats":  "human",
    # Add more as needed
}


def map_to_coarse(activity):
    activity_l = activity.lower()
    if activity_l in ACTIVITY_TO_CLASS:
        return ACTIVITY_TO_CLASS[activity_l]
    # Heuristic: anything containing 'walk', 'jump', 'stand', 'fall', 'sit', 'lie' is human
    for key in ["walk", "stand", "fall", "sit", "lie", "human", "box", "jump", "squat"]:
        if key in activity_l:
            return "human"
    return "unknown"


# ---- Step 5: build sliding-window tensor dataset ----

def build_window_dataset(root, output_npz, window_size=16, stride=4, max_targets=3):
    """Load RadHAR, convert to RD-03D-format targets, slide window, save."""
    frames, activities = load_dataset(root)
    print(f"\nLoaded {len(frames)} raw point-cloud frames")

    if len(frames) < window_size:
        raise ValueError(f"Need at least {window_size} frames, got {len(frames)}")

    # Convert each raw frame -> list of RD-03D targets
    print("Clustering and projecting frames to RD-03D format...")
    frame_targets = []
    for i, frame in enumerate(frames):
        centroids = cluster_to_centroids(frame, max_targets=max_targets)
        targets = project_to_rd03d(centroids)
        frame_targets.append(targets)
        if (i + 1) % 5000 == 0:
            print(f"  processed {i+1}/{len(frames)} frames")

    # Slide windows
    print("\nSliding windows...")
    rows = []
    for start in range(0, len(frame_targets) - window_size + 1, stride):
        window = frame_targets[start : start + window_size]
        window_activities = activities[start : start + window_size]

        # Tensor: (window_size, max_targets, 4)
        tensor = np.zeros((window_size, max_targets, 4), dtype=np.float32)
        mask = np.zeros((window_size, max_targets), dtype=np.float32)

        for t, frame_t in enumerate(window):
            for k, tgt in enumerate(frame_t[:max_targets]):
                tensor[t, k, 0] = tgt["d_m"]
                tensor[t, k, 1] = tgt["speed_mps"]
                tensor[t, k, 2] = tgt["x_m"]
                tensor[t, k, 3] = tgt["y_m"]
                mask[t, k] = 1.0

        # Window-level label: majority activity
        from collections import Counter
        major_activity = Counter(window_activities).most_common(1)[0][0]
        coarse_class = map_to_coarse(major_activity)

        rows.append({
            "tensor": tensor,
            "mask": mask,
            "activity": major_activity,
            "class": coarse_class,
        })

    tensors = np.stack([r["tensor"] for r in rows])
    masks = np.stack([r["mask"] for r in rows])
    activities_arr = np.array([r["activity"] for r in rows])
    classes_arr = np.array([r["class"] for r in rows])

    np.savez_compressed(
        output_npz,
        tensors=tensors,
        masks=masks,
        activities=activities_arr,
        classes=classes_arr,
    )

    print(f"\nSaved {len(rows)} windows -> {output_npz}")
    print(f"Tensor shape: {tensors.shape}, mask shape: {masks.shape}")
    unique, counts = np.unique(classes_arr, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")
    unique_act, counts_act = np.unique(activities_arr, return_counts=True)
    print(f"Activity distribution: {dict(zip(unique_act, counts_act))}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Path to RadHAR Train or Test folder")
    p.add_argument("--out", default="rd03d_dataset.npz")
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--stride", type=int, default=4)
    args = p.parse_args()

    build_window_dataset(args.root, args.out, window_size=args.window, stride=args.stride)
