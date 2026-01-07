#!/usr/bin/env python3
"""
make_dataset.py

Build ML-ready tabular datasets from Delphes ROOT files.

Responsibilities:
  - Read input ROOT paths from config/path.yaml
  - Select exactly one b1, one b2, and one c1 jet per event
  - Flatten event-level information into a tabular format
  - Write output datasets in CSV or Parquet format

Designed to scale to large ROOT inputs (O(100–300 GB)).

Inputs:
  Delphes ROOT files specified in config/path.yaml

Outputs:
  Tabular datasets in CSV or Parquet format

Examples:
  python src/make_dataset.py
  python src/make_dataset.py --format parquet --outdir ml_outputs
"""


from pathlib import Path
import argparse

from src.utils.config import load_paths
from src.jet_selection import main as run_jet_selection


# Helper: call jet_selection.py programmatically
def run_sample(
    files,
    output,
    label,
    require_exact=True,
    fmt="parquet",
):
    """
    Run jet selection on a list of ROOT files.
    """
    args = []

    for f in files:
        args += ["-i", str(f)]

    args += [
        "-o", str(output),
        "--label", str(label),
        "--require-exact",
    ]

    if fmt:
        args += ["--format", fmt]

    print(f"\n▶ Processing {len(files)} file(s)")
    print(f"  Output: {output}")
    print(f"  Label:  {label}")

    # Temporarily replace sys.argv for jet_selection.main()
    import sys
    old_argv = sys.argv
    sys.argv = ["jet_selection.py"] + args

    try:
        run_jet_selection()
    finally:
        sys.argv = old_argv


# --------------------------------------
# Main
# --------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build signal/background ML datasets.")
    ap.add_argument("--config", default="configs/path.yaml")
    ap.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default="parquet",
        help="Output format (default: parquet)",
    )
    ap.add_argument(
        "--outdir",
        default="root_outputs",
        help="Output directory (default: root_outputs/)",
    )
    args = ap.parse_args()

    paths = load_paths(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Signal samples
    for sig_name, files in paths["signal"].items():
        out = outdir / f"signal_{sig_name}.{args.format}"
        run_sample(
            files=files,
            output=out,
            label=1,
            fmt=args.format,
        )

    # Background samples
    for bkg_name, files in paths["background"].items():
        out = outdir / f"background_{bkg_name}.{args.format}"
        run_sample(
            files=files,
            output=out,
            label=0,
            fmt=args.format,
        )

    print("\n Dataset build complete.")


if __name__ == "__main__":
    main()

