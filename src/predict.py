#!/usr/bin/env python3
"""
predict.py

Inference-only script for frozen XGBoost models.

Reads:
  - model.ubj (preferred) or model.joblib
  - features.txt (ordered feature list)
  - threshold.json (decision threshold + metadata)

Inputs:
  - Parquet or CSV file with the required feature columns

Outputs:
  - parquet/csv with bdt_score and bdt_pred (score >= threshold)

Example:
  python src/predict.py --mass 200 \
    --model-dir ml_models/final \
    --input ml_outputs/splits/test_sig200.parquet \
    --split test 
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import re

try:
    import xgboost as xgb
except ImportError as e:
    raise SystemExit("xgboost is required for inference. Install: pip install xgboost") from e

try:
    import joblib
except ImportError:
    joblib = None

from src.utils.sanity import require_numeric, check_no_nan_inf, sanity_predictions

def _latest_final_version(final_root: Path, mass: int) -> Optional[int]:
    # dirs look like: sig200_v1, sig200_v2, ...
    pat = re.compile(rf"^sig{mass}_v(\d+)$")
    best = None
    for p in final_root.iterdir():
        if not p.is_dir():
            continue
        m = pat.match(p.name)
        if not m:
            continue
        v = int(m.group(1))
        best = v if best is None else max(best, v)
    return best # can be None

def read_features(path: Path) -> List[str]:
    feats = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not feats:
        raise ValueError(f"No features found in {path}")
    return feats


def load_threshold(path: Path) -> float:
    thr = json.loads(path.read_text()).get("threshold", None)
    if thr is None:
        raise KeyError(f"'threshold' not found in {path}")
    return float(thr)


def load_model(model_dir: Path):
    """
    Prefer model.ubj. Fall back to model.joblib.
    Returns a callable object that provides predict_proba or booster predict.
    """
    ubj_path = model_dir / "model.ubj"
    joblib_path = model_dir / "model.joblib"

    if ubj_path.exists():
        booster = xgb.Booster()
        booster.load_model(ubj_path.as_posix())

        def predict_scores(X: np.ndarray) -> np.ndarray:
            dmat = xgb.DMatrix(X)
            p = booster.predict(dmat)
            return p.astype(np.float32)

        return "ubj", ubj_path, predict_scores

    if joblib_path.exists():
        if joblib is None:
            raise SystemExit("joblib not available but model.joblib exists. Install: pip install joblib")
        model = joblib.load(joblib_path)

        def predict_scores(X: np.ndarray) -> np.ndarray:
            # XGBClassifier exposes predict_proba
            p = model.predict_proba(X)[:, 1]
            return p.astype(np.float32)

        return "joblib", joblib_path, predict_scores

    raise FileNotFoundError(f"No model found in {model_dir} (expected model.ubj or model.joblib)")

def load_split_meta(mass: int, splits_dir: Path = Path("ml_outputs/splits")) -> Dict[str, Any]:
    """
    Load split metadata produced by prepare_ml.py / stratified split writer.
    Expected file: ml_outputs/splits/split_sig{MASS}.meta.json
    Returns {} if not found (so predict.py still works standalone).
    """
    meta_path = splits_dir / f"split_sig{mass}.meta.json"
    if not meta_path.exists():
        print(f"[WARN] Split meta not found: {meta_path} (skipping weighted_frac_*)")
        return {}
    try:
        return json.loads(meta_path.read_text())
    except Exception as e:
        print(f"[WARN] Failed to read split meta: {meta_path} ({e})")
        return {}

def read_input(path: Path, fmt: str) -> pd.DataFrame:
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported format: {fmt}. Choose parquet or csv.")


def write_output(df: pd.DataFrame, path: Path, fmt: str):
    if fmt == "parquet":
        df.to_parquet(path, index=False)
        return
    if fmt == "csv":
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported format: {fmt}. Choose parquet or csv.")


# --------------------------------------
# Main
# --------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="ml_models/final", help="final models directory")
    ap.add_argument("--mass",type=int, default=200, help="Signal mass tag, e.g. 200")
    ap.add_argument("--ver", default="latest",
                    help=("Frozen model version: integer (e.g. 1) or 'latest'. "
                          "If no frozen model exists, fall back to working dir ml_models/sig{mass}."))
    ap.add_argument("--input", required=True, help="Input parquet/csv with feature columns")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"], help="Which infer file to generate")
    ap.add_argument("--out-name", default=None,
                    help="Override output filename (default: preds_infer_{split}.parquet)")
    ap.add_argument("--outdir-name", default=None,
                    help="Optional output directory. If not set, outputs go under model_dir")
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet", help="Input/output format")
    ap.add_argument("--keep-cols", nargs="*", default=["target", "sample", "sample_weight", "gen_weight"],
                    help="Optional columns to keep if present")
    ap.add_argument("--no-pred", action="store_true", help="Do not create hard label bdt_pred")
    args = ap.parse_args()

    base_dir = Path(args.model_dir)
    mass = int(args.mass)
    tag_base = f"sig{mass}"

    ver = str(args.ver).strip().lower()

    if ver == "latest":
        version = _latest_final_version(base_dir, mass)
        if version is None:
            # Fallback to working directory
            model_dir = Path("ml_models") / tag_base
            print(f"[predict] No frozen model found. Using working dir: {model_dir}")
        else:
            model_dir = base_dir / f"{tag_base}_v{version}"
            print(f"[predict] Using latest frozen model: {model_dir}")
    else:
        # explicit version
        try:
            v = int(ver)
        except ValueError as e:
            raise SystemExit(f"--ver must be an integer or 'latest' (got: {args.ver!r})") from e

        model_dir = base_dir / f"{tag_base}_v{v}"
        print(f"[predict] Using specified frozen model: {model_dir}")

    in_path = Path(args.input)

    if args.outdir_name is not None:
        out_dir = Path(args.outdir_name)
    else:
        out_dir = model_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    out_name  = args.out_name or f"preds_infer_{args.split}.{args.format}"
    out_path = out_dir / out_name

    features_path = model_dir / "features.txt"
    threshold_path = model_dir / "threshold.json"

    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}")
    if not threshold_path.exists():
        raise FileNotFoundError(f"Missing {threshold_path}")

    features = read_features(features_path)
    thr = load_threshold(threshold_path)

    model_kind, model_path, predict_scores = load_model(model_dir)
    print(f"[INFO] Loaded model ({model_kind}): {model_path}")
    print(f"[INFO] Loaded features: {features_path} ({len(features)} features)")
    print(f"[INFO] Loaded threshold: {threshold_path} (thr={thr:.6f})")

    df = read_input(in_path, args.format)
    # ---- schema sanity ----
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required features in input: {missing}")
    # ---- value sanity ----
    require_numeric(df, features, name="predict_input")
    check_no_nan_inf(df, features, name="predict_input")

    X = df[features].to_numpy(dtype=np.float32)
    scores = predict_scores(X)

    # Build output with a minimal contract:
    keep = [c for c in args.keep_cols if c in df.columns]
    out = df[keep].copy() if keep else pd.DataFrame(index=df.index)
    out["bdt_score"] = scores

    if not args.no_pred:
        out["bdt_pred"] = (out["bdt_score"] >= thr).astype(np.int8)

    # ---- score sanity ----
    sanity_predictions(out, score_col="bdt_score", name="predict_output")

    # Helpful metadata 
    pred_path = out_path
    if pred_path.suffix.lower() not in [".parquet", ".csv"]:
        pred_path = pred_path.with_suffix(f".{args.format}")
    write_output(out, pred_path, args.format)

    meta = {
        "model_dir": str(model_dir),
        "model_kind": model_kind,
        "model_path": str(model_path),
        "threshold": float(thr),
        "features_path": str(features_path),
        "threshold_path": str(threshold_path),
        "n_rows": int(len(out)),
        "keep_cols": [c for c in args.keep_cols if c in df.columns],
        "features": features,
    }

    is_split_input = "splits" in in_path.parts
    if not is_split_input:
        print(f"[WARN] Input file is not from splits/: {in_path}")
        print("[WARN] weighted_frac_* will be taken from split meta if available, otherwise left as None")

    split_meta = load_split_meta(mass)

    weighted_frac_current = None
    if args.split == "val":
        weighted_frac_current = split_meta.get("weighted_frac_val", None)
    elif args.split == "test":
        weighted_frac_current = split_meta.get("weighted_frac_test", None)

    meta["input_is_split"] = is_split_input
    meta["weighted_frac_current_split"] = weighted_frac_current

    meta_path = pred_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"[OK] Wrote: {pred_path}  (rows={len(out)})")
    print(f"[OK] Wrote: {meta_path}")


if __name__ == "__main__":
    main()

