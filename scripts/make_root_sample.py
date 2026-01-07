#!/usr/bin/env python3
"""
make_root_sample.py

Create small, representative ROOT samples from large Delphes ROOT inputs
for rapid testing and development of the ML pipeline.

This script reduces full-statistics ROOT files (O(100–300 GB)) to compact
sample datasets suitable for debugging, feature engineering, and workflow
validation.

Inputs:
  raw Delphes ROOT files in rawdata/

Outputs:
  sample ROOT files saved in sampledata/
    - ~2000 events per signal sample
    - ~3000 events per background sample

Example:
  python scripts/make_root_sample.py
"""

from __future__ import annotations
import subprocess
from pathlib import Path

from src.utils.config import load_paths

N_SIG = 2000
N_BKG = 3000

def run_cut(exe: Path, infile: Path, outfile: Path, n: int) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(exe), str(infile), str(outfile), str(n), "Delphes"]
    subprocess.run(cmd, check=True)

def main():
    paths = load_paths("configs/path.yaml")
    outdir = Path("sampledata")
    exe = Path("scripts/cut_delphes_tree")

    if not exe.exists():
        raise SystemExit(
            f"Missing {exe}. Build it first:\n"
            "g++ -O3 scripts/cut_delphes_tree.cpp -o scripts/cut_delphes_tree $(root-config --cflags --libs)"
        )

    # signals
    for sig_key, files in paths["signal"].items():
        for infile in files:
            infile = Path(infile)
            outfile = outdir / infile.name  # keep same name
            print(f"{infile.name} to {outfile}  (keep {N_SIG})")
            run_cut(exe, infile, outfile, N_SIG)

    # backgrounds
    for bkg_key, files in paths["background"].items():
        for infile in files:
            infile = Path(infile)
            outfile = outdir / infile.name
            print(f"{infile.name} to {outfile}  (keep {N_BKG})")
            run_cut(exe, infile, outfile, N_BKG)

if __name__ == "__main__":
    main()

