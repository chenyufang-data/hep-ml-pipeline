#!/usr/bin/env python3
"""
run_optimal_ablation.py

Select the optimal feature subset based on drop-1 ablation results and
retrain the final model using the reduced feature set. Then generate
standard diagnostic plots for the optimized model.

Workflow:
  1. Read drop-1 ablation summary
  2. Identify features whose removal improves or least degrades performance
  3. Retrain model with the optimized feature set
  4. Run diagnostic plotting on the final model

Inputs (default):
  ml_ablation/sig{MASS}/drop1_summary.csv

Outputs:
  ml_models/final/sig{MASS}/
    - model.ubj
    - metrics.json
    - diagnostic plots

Example:
  python src/run_optimal_ablation.py --mass 200 --mode drop1
"""

import pandas as pd
import subprocess
import argparse
from pathlib import Path
import sys
import os

# --------------------------------------
# Main
# --------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Run optimal BDT training based on ablation results.")
    ap.add_argument("--csvdir", default="ml_ablation", help="Directory containing greedy_summary.csv")
    ap.add_argument("--mass", type=int, default=200, help="Signal mass tag, e.g. 200")
    ap.add_argument("--mode", choices=["drop1", "greedy"], default="drop1",help="Ablation mode choice, e.g. drop1 or greedy")
    ap.add_argument("--outdir", default="ml_models/work", help="Final BDT train output directory")
    args = ap.parse_args()

    # Define Paths
    csv_path = Path(args.csvdir) / f"sig{args.mass}" / f"{args.mode}_summary.csv"
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for File
    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    # Parse CSV to find optimal feature set
    df = pd.read_csv(csv_path)
    
    if args.mode == "greedy":
        # Identify the step with the highest new_val_Z
        best_row = df.loc[df['new_val_Z'].idxmax()]
        best_step = best_row['step']
        
        # ---- guard against non-improving step-0 ablation ----
        if best_step == 0:
            if best_row["new_val_Z"] <= best_row["base_val_Z"]:
                print(
                    f"[INFO] Step-0 ablation does not improve Z: "
                    f"new={best_row['new_val_Z']:.4f}, "
                    f"base={best_row['base_val_Z']:.4f}"
                )
                print("[INFO] Using baseline feature set (no drops).")
                best_step = -1

        # Collect all features dropped up to and including that best step
        features_to_drop = (
            df[df['step'] <= best_step]['removed'].tolist()
            if best_step >= 0 else []
        )

        print(f"--- Ablation Results for sig{args.mass} ---")
        print(f"Best val_Z ({best_row['new_val_Z']:.4f}) found at step {best_step}")
        print(f"Dropping {len(features_to_drop)} features: {features_to_drop}")
    else:
        # Filter first
        positive_dz = df[df['d_val_Z'] > 0]

        if not positive_dz.empty:
            # find the row with the maximum d_val_Z
            best_row = positive_dz.loc[positive_dz['d_val_Z'].idxmax()]
            
            # find 'dropped' 
            features_to_drop = [best_row['dropped']]

            print(f"--- Ablation Results for sig{args.mass} ---")
            print(f"Best val_Z ({best_row['val_Z']:.4f}) found by dropping {best_row['dropped']}")
        else:
            best_row = None
            features_to_drop = []
            print(f"--- No positive d_val_Z found for sig{args.mass} ---")

    # 4. Construct Command
    # We include --outdir and the list of features to drop
    cmd = [
        sys.executable, "-m", "src.train_bdt",
        "--mass", str(args.mass),
        "--outdir", output_dir
    ]
    
    if features_to_drop:
        cmd.append("--drop-features")
        cmd.extend(features_to_drop)

    # 5. Execute
    print(f"\nRunning: {' '.join(map(str,cmd))}\n")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nTraining script failed with exit code {e.returncode}")
        sys.exit(e.returncode)

    # 5. Plot
    bdtplot = [
        sys.executable, "-m", "src.plot_bdt_diagnostics",
        "--modeldir", output_dir,
        "--mass", str(args.mass),
    ]
    print(f"\nRunning: {' '.join(map(str,bdtplot))}\n")
    try:
        subprocess.run(bdtplot, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nPlot failed with exit code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
