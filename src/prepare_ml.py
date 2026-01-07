#!/usr/bin/env python3
"""
prepare_ml.py

Prepare ML-ready datasets from the parquet outputs produced by make_dataset.py.
Merges signal and background samples, performs basic sanity checks, and
optionally writes train/validation/test splits.

Workflow:
  - Combine signal and background samples for a given mass point
  - Generate a sanity report (event counts, weights, basic statistics)
  - Optionally split into train/val/test sets (stratified by label)

Inputs:
  Parquet files in the input directory produced by make_dataset.py

Outputs:
  ml_outputs/sig{MASS}.parquet
  ml_outputs/splits/
    - train_sig{MASS}.parquet
    - val_sig{MASS}.parquet
    - test_sig{MASS}.parquet
    - split_sig{MASS}.meta.json

Examples:
  python src/prepare_ml.py --write-splits

  python src/prepare_ml.py \
    --indir root_outputs \
    --outdir ml_outputs \
    --mass 200 \
    --backgrounds bbj ccj bkgbbc \
    --write-splits \
    --test-size 0.1 --val-size 0.1
"""

from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from src.utils.sanity import sanity_prepared_ml
import json


DEFAULT_FEATURES = [
    "ptb1", "etab1",
    "ptb2", "etab2",
    "ptc1", "etac1",
    "mbc1", "mbc2",
    "dr13", "dr23",
    "ht","mcbb",
    "mbb","ratio_ptcb",
    "dr12",
]
REQUIRED_META = ["label", "weight", "xs"]


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_parquet(path)

def _read_meta(indir: Path, stem: str) -> dict:
    """
    Read meta json written by jet_selection.py.
    stem examples:
      - "signal_sig_200"
      - "background_bbj"
      - "background_ccj"
      - "background_bbc"
    """
    meta_path = indir / f"{stem}.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta file: {meta_path}")
    return json.loads(meta_path.read_text())

def split_fracs_weighted(df_all, df_split, y_col="target", w_col="sample_weight"):
    S_all = float(df_all.loc[df_all[y_col]==1, w_col].sum())
    B_all = float(df_all.loc[df_all[y_col]==0, w_col].sum())
    S_sp  = float(df_split.loc[df_split[y_col]==1, w_col].sum())
    B_sp  = float(df_split.loc[df_split[y_col]==0, w_col].sum())
    return {"f_sig": S_sp / S_all, "f_bkg": B_sp / B_all}

def _clean_df(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    # keep only needed columns if they exist
    needed = REQUIRED_META + features
    missing = [c for c in REQUIRED_META if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in dataframe with columns={list(df.columns)}")

    # create missing feature columns (if any) as NaN so concat still works
    for c in features:
        if c not in df.columns:
            df[c] = np.nan

    df = df[needed].copy()

    # replace inf with NaN then drop rows with NaN in features
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=features, inplace=True)

    # make sure dtypes are numeric where possible
    for c in features + ["weight"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=features + ["weight"], inplace=True)

    # labels should be int 0/1
    df["label"] = df["label"].astype(int)

    return df


def _stratified_split(df: pd.DataFrame, test_size: float, val_size: float, seed: int):
    """
    Split into train/val/test with stratification on 'target'.
    val_size is fraction of *full dataset* (not of train).
    """
    if test_size < 0 or val_size < 0 or (test_size + val_size) >= 1.0:
        raise ValueError(f"Invalid split sizes: test={test_size}, val={val_size}. Need test+val < 1.")

    rng = np.random.default_rng(seed)

    # indices per class
    idx_sig = df.index[df["target"] == 1].to_numpy()
    idx_bkg = df.index[df["target"] == 0].to_numpy()

    def split_class(idxs):
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = int(round(test_size * n))
        n_val = int(round(val_size * n))
        test = idxs[:n_test]
        val = idxs[n_test:n_test + n_val]
        train = idxs[n_test + n_val:]
        return train, val, test

    tr_s, va_s, te_s = split_class(idx_sig.copy())
    tr_b, va_b, te_b = split_class(idx_bkg.copy())

    train_idx = np.concatenate([tr_s, tr_b])
    val_idx   = np.concatenate([va_s, va_b])
    test_idx  = np.concatenate([te_s, te_b])

    # shuffle each split
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return df.loc[train_idx].reset_index(drop=True), df.loc[val_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)


def build_one_dataset(indir: Path, outdir: Path, sig_mass: str, backgrounds: list[str], features: list[str], lumi: float) -> pd.DataFrame:
    # load per-sample meta (PRE-SELECTION normalizations)
    sig_meta = _read_meta(indir, f"signal_sig_{sig_mass}")
    sig_sumw_all = float(sig_meta["sum_w_all"])

    sig_files = sorted( indir.glob(f"signal_sig_{sig_mass}.part*.parquet"))
    if not sig_files:
        raise FileNotFoundError(f"No signal parquet files for mass {sig_mass}")

    sig = pd.concat(
        ( _clean_df(_read_parquet(p), features) for p in sig_files),
        ignore_index=True,
    )
    sig["sample"] = f"sig_{sig_mass}"

    bkg_meta = {}
    bkg_sumw_all = {}
    bkg_frames = []
    for b in backgrounds:
        m = _read_meta(indir, f"background_{b}")
        bkg_meta[b] = m
        bkg_sumw_all[b] = float(m["sum_w_all"])

        bkg_files = sorted(indir.glob(f"background_{b}.part*.parquet"))
        if not bkg_files:
            raise FileNotFoundError(f"No background parquet files for {b}")
        
        bkg = pd.concat(
            (_clean_df(_read_parquet(p), features) for p in bkg_files),
            ignore_index=True,
            )
        bkg["sample"] = f"bkg_{b}"
        bkg_frames.append(bkg)

    df = pd.concat([sig] + bkg_frames, ignore_index=True)

    # rename for clarity
    df = df.rename(columns={
        "label": "target",
        "weight": "gen_weight",
        "xs": "evt_xs",
    })

    # attach a per-row normalization denominator based on sample
    df["sum_w_all"] = np.nan
    df.loc[df["sample"] == f"sig_{sig_mass}", "sum_w_all"] = sig_sumw_all
    for b in backgrounds:
        df.loc[df["sample"] == f"bkg_{b}", "sum_w_all"] = bkg_sumw_all[b]

    if df["sum_w_all"].isna().any():
        bad = df.loc[df["sum_w_all"].isna(), "sample"].unique()
        raise ValueError(f"Missing sum_w_all for samples: {bad}")

    # physics weight used at a choosen lumi for ML evaluation:
    lumi = float(lumi)
    df["sample_weight"] = lumi * (2.0 * df["gen_weight"] * df["evt_xs"]) / df["sum_w_all"]

    # ---- HARD sanity checks (per sample) ----
    report = sanity_prepared_ml(
        df,
        name=f"sig{sig_mass}_prepared",
        lumi=lumi,
    )

    # ---- SAVE PER-SAMPLE REPORT ----
    sanity_root = outdir / "sanity_reports"
    sanity_dir = sanity_root / f"sig{sig_mass}"
    sanity_dir.mkdir(parents=True, exist_ok=True)

    (sanity_dir / "sanity_prepared.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    # ---- sanity prints ----
    counts = df["target"].value_counts(dropna=False)
    S = float(df.loc[df["target"] == 1, "sample_weight"].sum())
    B = float(df.loc[df["target"] == 0, "sample_weight"].sum())

    # Effective cross sections AFTER jet_selection, per process:
    # sigma_after = sum(sample_weight)/lumi
    xs_after_sig = S / lumi if lumi > 0 else 0.0
    xs_after_bkg = B / lumi if lumi > 0 else 0.0

    print(f"\nBuilt dataset for sig_{sig_mass}: shape={df.shape}")
    print("Target counts:\n", counts.to_string())
    print(f"Sum sample_weight (yields at L={lumi}):  S={S:.6g}, B={B:.6g}")
    print(f"Effective xs after jet_selection:        S={xs_after_sig:.6g}, B={xs_after_bkg:.6g}")

    return df


# --------------------------------------
# Main
# --------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="root_outputs", help="Directory containing signal_*.parquet and background_*.parquet")
    ap.add_argument("--outdir", default="ml_outputs", help="Where to write combined ML datasets")
    ap.add_argument("--mass", nargs="+", default=["200"], help="Signal mass tag, e.g. 200")
    ap.add_argument("--backgrounds", nargs="+", default=["bbj", "ccj", "bbc"], help="Background names, e.g. bbj ccj bbc")
    ap.add_argument("--features", nargs="+", default=DEFAULT_FEATURES, help="Feature columns to keep")
    ap.add_argument("--extra-features", nargs="+",default=[], help="Extra features to append to base list")
    ap.add_argument("--write-splits", action="store_true", help="Also write train/val/test splits for each dataset")
    ap.add_argument("--test-size", type=float, default=0.1)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lumi", type=float, default=3000.0, help="Integrated luminosity multiplier for sample_weight (default=3000)")

    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    features = args.features + args.extra_features

    for sig_mass in args.mass:
        df = build_one_dataset(indir, outdir, sig_mass, args.backgrounds, features, args.lumi)

        out_all = outdir / f"dataset_sig{sig_mass}_vs_bkg.parquet"
        df.to_parquet(out_all, index=False)
        print(f"Saved: {out_all}")

        if args.write_splits:
            train, val, test = _stratified_split(df, args.test_size, args.val_size, args.seed)
            (outdir / "splits").mkdir(parents=True, exist_ok=True)

            train_path = outdir / "splits" / f"train_sig{sig_mass}.parquet"
            val_path   = outdir / "splits" / f"val_sig{sig_mass}.parquet"
            test_path  = outdir / "splits" / f"test_sig{sig_mass}.parquet"

            train.to_parquet(train_path, index=False)
            val.to_parquet(val_path, index=False)
            test.to_parquet(test_path, index=False)

            counts = {
                "n_all": int(len(df)),
                "n_train": int(len(train)),
                "n_val": int(len(val)),
                "n_test": int(len(test)),
                "frac_train": float(len(train) / len(df)),
                "frac_val": float(len(val) / len(df)),
                "frac_test": float(len(test) / len(df)),
            }

            split_meta = {
                "test_size_arg": args.test_size,
                "val_size_arg": args.val_size,
                "seed": args.seed,
                "counts": counts,
                "weighted_frac_val": split_fracs_weighted(df, val),
                "weighted_frac_test": split_fracs_weighted(df, test),
            }
            (outdir / "splits" / f"split_sig{sig_mass}.meta.json").write_text(json.dumps(split_meta, indent=2) + "\n")

            print(f"Saved splits: {train_path}, {val_path}, {test_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

