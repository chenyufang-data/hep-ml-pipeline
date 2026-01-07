#!/usr/bin/env python3
"""
freeze_final.py

Freeze the final optimized model by versioning the selected feature set
and decision threshold. This creates an immutable snapshot of the model
for reproducibility and deployment.

Workflow:
  - Read final model artifacts from ml_models/final/sig{MASS}/
  - Assign a version tag (e.g. v1, v2, ...)
  - Write a versioned copy to ml_models/final/sig{MASS}_v{x}/

Inputs (default):
  ml_models/final/sig{MASS}/

Outputs:
  ml_models/final/sig{MASS}_v{x}/
    - model.ubj 
    - model.joblib
    - metrics.json
    - features.txt
    - threshold.json

Example:
  python src/freeze_final.py --mass 200
"""

from pathlib import Path
import json
import argparse
import re
import shutil
import os

def next_version(base: Path, tag: str) -> Path:
    """
    Find next available version directory:
      sig200_v1, sig200_v2, ...
    """
    pattern = re.compile(rf"{re.escape(tag)}_v(\d+)")
    versions = []
    for p in base.iterdir():
        if p.is_dir():
            m = pattern.fullmatch(p.name)
            if m:
                versions.append(int(m.group(1)))
    v = 1 if not versions else max(versions) + 1
    return base / f"{tag}_v{v}"

# --------------------------------------
# Main
# --------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finaldir", default="ml_models/final",
                    help="directory for final model releases")
    ap.add_argument("--mass", type=int, default=200, help="Signal mass tag, e.g. 200")
    ap.add_argument("--use-threshold-from", choices=["val", "test"],
                    default="val",
                    help="Which split provides the frozen threshold")
    args = ap.parse_args()

    base_dir = Path(args.finaldir)   
    base_dir.mkdir(parents=True, exist_ok=True)
    src_dir  = base_dir.parent / f"work/sig{args.mass}" 
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    metrics_path = src_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json in {src_dir}")

    metrics = json.loads(metrics_path.read_text())

    if "features" not in metrics:
        raise KeyError("metrics.json missing 'features'")
    if args.use_threshold_from not in metrics:
        raise KeyError(f"metrics.json missing split '{args.use_threshold_from}'")
    if "weighted_significance_scan" not in metrics[args.use_threshold_from]:
        raise KeyError("Missing weighted_significance_scan in metrics")

    scan = metrics[args.use_threshold_from]["weighted_significance_scan"]
    required = ["best_thr", "best_Z", "best_S", "best_B"]
    for k in required:
        if k not in scan:
            raise KeyError(f"weighted_significance_scan missing '{k}'")

    # create versioned release dir
    tag = f"sig{args.mass}"
    final_dir = next_version(base_dir, tag)
    final_dir.mkdir(parents=True, exist_ok=False)

    # copy model binary
    # files to copy
    files_to_copy = [
        "model.ubj",
        "model.joblib",
    ]
    for fname in files_to_copy:
        src = src_dir / fname
        if src.exists():
            print(f"[freeze] Copy file: {src.name}")
            shutil.copy2(src, final_dir / fname)

    # directories to copy 
    dirs_to_copy = [
        "plots",
    ]

    for dname in dirs_to_copy:
        src = src_dir / dname
        if src.exists() and src.is_dir():
            print(f"[freeze] Copy dir:  {dname}/")
            shutil.copytree(src, final_dir / dname)

    # ---- freeze features ----
    features = metrics["features"]
    (final_dir / "features.txt").write_text(
        "\n".join(features) + "\n"
    )

    # ---- freeze threshold ----
    threshold = {
        "threshold": float(scan["best_thr"]),
        "split_used": args.use_threshold_from,
        "Z_Asimov": float(scan["best_Z"]),
        "S": float(scan["best_S"]),
        "B": float(scan["best_B"]),
        "method": scan.get("method", "argmax"),
        "constraints": scan.get("constraints", {}),
        "raw_argmax": scan.get("raw_argmax", {
            "threshold": float(scan["best_thr"]),
            "Z": float(scan["best_Z"]),
        }),
        "note": (
            "Threshold chosen from validation set using a robust plateau-based "
            "Asimov Z scan with minimum unweighted event constraints."
            if args.use_threshold_from == "val"
            else
            "Threshold chosen from test set for reference only; "
            "validation-based threshold is preferred."
        )
    }

    (final_dir / "threshold.json").write_text(
        json.dumps(threshold, indent=2) + "\n"
    )

    # copy metrics.json for traceability
    (final_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )

    required = ["features.txt", "threshold.json"]
    missing = [f for f in required if not (final_dir / f).exists()]
    if missing:
        raise RuntimeError(f"Freeze incomplete; missing {missing} in {final_dir}")

    print(f"[OK] Created release: {final_dir}")
    print(f"[OK] Frozen features  -> {final_dir/'features.txt'}")
    print(f"[OK] Frozen threshold -> {final_dir/'threshold.json'}")

    latest_txt = base_dir / f"LATEST_sig{args.mass}.txt"
    latest_txt.write_text(f"{final_dir.name}\n")
    
    print(f"[freeze] Wrote latest version directory: {latest_txt}")

    try:
        link_ptr = base_dir / f"sig{args.mass}"

        # Remove old pointer dir or symlink
        if os.path.lexists(link_ptr):
            if link_ptr.is_symlink() or link_ptr.is_file():
                link_ptr.unlink()
            elif link_ptr.is_dir():
                shutil.rmtree(link_ptr)

        # Create symlink
        rel_target = os.path.relpath(final_dir, start=link_ptr.parent)
        link_ptr.symlink_to(rel_target, target_is_directory=True)
        print(f"[freeze] Updated symlink pointer: {link_ptr} -> {rel_target}")
    except Exception as e:
        print(f"[freeze] Symlink not created (this is OK on Windows): {e}")
        print(f"[freeze] Use LATEST.txt to resolve latest version instead.")


if __name__ == "__main__":
    main()

