"""
Run the entire Phase 2 pipeline end-to-end.

Steps:
  1. Convert ICLegend dataset -> rd03d-format windows
  2. Generate synthetic vehicle samples
  3. Generate synthetic unknown/clutter samples
  4. Merge all three into combined_windows.npz
  5. Compute 72-dim features
  6. Train MLP
  7. (Optional) Train 1D-CNN
  8. Quantize to INT8 TFLite
  9. Export C array and metadata headers
 10. Run Layer 1 offline verification

Usage (after downloading ICLegend dataset):
    python run_phase2_pipeline.py \
        --iclegend /path/to/iclegend/dataset \
        --out-dir .
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    """Run a subprocess, abort on failure."""
    print(f"\n>>> {' '.join(cmd)}")
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        print(f"FAILED with code {r.returncode}")
        sys.exit(r.returncode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iclegend", required=False, default=None,
                   help="Path to ICLegend dataset root (skip with --synthetic-only)")
    p.add_argument("--synthetic-only", action="store_true",
                   help="Skip ICLegend, generate humans/vehicles/unknown synthetically")
    p.add_argument("--out-dir", default=".", help="Phase2 root directory")
    p.add_argument("--n-synthetic-vehicle", type=int, default=2000)
    p.add_argument("--n-synthetic-unknown", type=int, default=2000)
    p.add_argument("--n-synthetic-human", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--skip-1dcnn", action="store_true",
                   help="Skip the 1D-CNN training (saves ~5 min)")
    args = p.parse_args()

    if not args.synthetic_only and not args.iclegend:
        p.error("Either --iclegend or --synthetic-only is required")

    root = Path(args.out_dir).resolve()
    data_dir  = root / "data"
    model_dir = root / "models"
    scripts   = root / "scripts"
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    if args.synthetic_only:
        # ---------- 1. Synthetic humans (stand-in until ICLegend arrives) ----------
        print("\n========== Step 1: Generate synthetic humans ==========")
        run([py, str(scripts / "generate_synthetic_humans.py"),
             "--n-samples", str(args.n_synthetic_human),
             "--out", str(data_dir / "synthetic_humans.npz")])

        # ---------- 2. Synthetic vehicles ----------
        print("\n========== Step 2: Generate synthetic vehicles ==========")
        run([py, str(scripts / "generate_synthetic_vehicles.py"),
             "--n-samples", str(args.n_synthetic_vehicle),
             "--out", str(data_dir / "synthetic_vehicles.npz")])

        # ---------- 3. Synthetic unknown ----------
        print("\n========== Step 3: Generate synthetic unknown ==========")
        run([py, str(scripts / "generate_synthetic_unknown.py"),
             "--n-samples", str(args.n_synthetic_unknown),
             "--out", str(data_dir / "synthetic_unknown.npz")])

        # ---------- 4. Merge all three ----------
        print("\n========== Step 4: Merge window datasets ==========")
        run([py, str(scripts / "merge_windows.py"),
             "--in", str(data_dir / "synthetic_humans.npz"),
                  str(data_dir / "synthetic_vehicles.npz"),
                  str(data_dir / "synthetic_unknown.npz"),
             "--out", str(data_dir / "combined_windows.npz")])
    else:
        # ---------- 1. Convert ICLegend ----------
        print("\n========== Step 1: Convert ICLegend -> RD-03D format ==========")
        run([py, str(scripts / "convert_iclegend_to_rd03d.py"),
             "--root", args.iclegend,
             "--out", str(data_dir / "rd03d_dataset.npz"),
             "--window", "16", "--stride", "4"])

        # ---------- 2. Synthetic vehicles ----------
        print("\n========== Step 2: Generate synthetic vehicles ==========")
        run([py, str(scripts / "generate_synthetic_vehicles.py"),
             "--n-samples", str(args.n_synthetic_vehicle),
             "--out", str(data_dir / "synthetic_vehicles.npz")])

        # ---------- 3. Synthetic unknown ----------
        print("\n========== Step 3: Generate synthetic unknown ==========")
        run([py, str(scripts / "generate_synthetic_unknown.py"),
             "--n-samples", str(args.n_synthetic_unknown),
             "--out", str(data_dir / "synthetic_unknown.npz")])

        # ---------- 4. Merge windows ----------
        print("\n========== Step 4: Merge window datasets ==========")
        run([py, str(scripts / "merge_windows.py"),
             "--in", str(data_dir / "rd03d_dataset.npz"),
                  str(data_dir / "synthetic_vehicles.npz"),
                  str(data_dir / "synthetic_unknown.npz"),
             "--out", str(data_dir / "combined_windows.npz")])

    # ---------- 5. Compute features ----------
    print("\n========== Step 5: Compute 72-dim features ==========")
    run([py, str(scripts / "compute_features.py"),
         "--in", str(data_dir / "combined_windows.npz"),
         "--out", str(data_dir / "combined_features.npz")])

    # ---------- 6. Train MLP ----------
    print("\n========== Step 6: Train MLP ==========")
    run([py, str(scripts / "train_mlp.py"),
         "--data", str(data_dir / "combined_features.npz"),
         "--epochs", str(args.epochs),
         "--out", str(model_dir)])

    # ---------- 7. Train 1D-CNN (optional) ----------
    if not args.skip_1dcnn:
        print("\n========== Step 7: Train 1D-CNN ==========")
        run([py, str(scripts / "train_1dcnn.py"),
             "--data", str(data_dir / "combined_windows.npz"),
             "--epochs", str(args.epochs),
             "--out", str(model_dir)])

    # ---------- 8. Quantize and export ----------
    print("\n========== Step 8: Quantize to INT8, export C array ==========")
    run([py, str(scripts / "quantize_export.py"),
         "--keras", str(model_dir / "model.keras"),
         "--norm",  str(model_dir / "norm_params.npz"),
         "--data",  str(data_dir / "combined_features.npz"),
         "--out",   str(model_dir)])

    # ---------- 9. Layer 1 verification ----------
    print("\n========== Step 9: Layer 1 offline verification ==========")
    run([py, str(scripts / "test_model_offline.py"),
         "--model", str(model_dir / "model_int8.tflite"),
         "--data",  str(data_dir / "combined_features.npz"),
         "--norm",  str(model_dir / "norm_params.npz"),
         "--n-test", "500"])

    print("\n========== Phase 2 complete ==========")
    print(f"Outputs:")
    print(f"  - {data_dir}/combined_windows.npz")
    print(f"  - {data_dir}/combined_features.npz")
    print(f"  - {model_dir}/model.keras")
    print(f"  - {model_dir}/model_int8.tflite")
    print(f"  - {model_dir}/model_data.cc")
    print(f"  - {model_dir}/model_metadata.h")
    print(f"  - {model_dir}/norm_params.npz")


if __name__ == "__main__":
    main()
