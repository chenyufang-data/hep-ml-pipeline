#!/usr/bin/env python3
"""
train_bdt.py

Train an XGBoost BDT on prepared parquet datasets and evaluate performance
using physics-weighted metrics.

Design principles:
  - Model is trained unweighted (or with optional class balancing)
  - All physics metrics are evaluated using sample_weight
  - Validation set is used for threshold selection and early stopping

Inputs (default):
  ml_outputs/splits/{train,val,test}_sig{MASS}.parquet

Each input parquet is expected to contain:
  - target (0 = background, 1 = signal)
  - feature columns (e.g. ptb1, etab1, ..., dr23)
  - sample_weight (physics weight for evaluation)
  - sample (optional string label, e.g. 'sig_200', 'bkg_bbj')

Outputs:
  ml_models/sig{MASS}/
    - model.ubj
    - model.joblib
    - metrics.json
    - preds_{train,val,test}.parquet

Example:
  python src/train_bdt.py --mass 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError as e:
    raise SystemExit("xgboost is required. Install: pip install xgboost") from e

try:
    from sklearn.metrics import roc_auc_score
except ImportError as e:
    raise SystemExit("scikit-learn is required. Install: pip install scikit-learn") from e

from utils.sanity import sanity_prepared_ml

DEFAULT_FEATURES = [
    "ptb1", "etab1", "ptb2", "etab2",
    "ptc1", "etac1",
    "mbc1", "mbc2",
    "dr13", "dr23",
    "ht","mcbb",
    "mbb","ratio_ptcb",
    "dr12",
]


# ------------------------------------------------------------
# IO
# ------------------------------------------------------------
def load_split(split_path: Path) -> pd.DataFrame:
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    df = pd.read_parquet(split_path)
    if "target" not in df.columns:
        raise KeyError(f"'target' column not found in {split_path}")
    return df

def split_fracs_by_class(df_all, df_split, y_col="target", w_col="sample_weight") -> tuple[float, float]:
    S_all = float(df_all.loc[df_all[y_col] == 1, w_col].sum())
    B_all = float(df_all.loc[df_all[y_col] == 0, w_col].sum())
    S_sp  = float(df_split.loc[df_split[y_col] == 1, w_col].sum())
    B_sp  = float(df_split.loc[df_split[y_col] == 0, w_col].sum())

    if S_all <= 0 or B_all <= 0:
        raise ValueError(f"Bad totals: S_all={S_all}, B_all={B_all}")
    if S_sp <= 0 or B_sp <= 0:
        raise ValueError(f"Bad splits: S_sp{S_sp}, B_sp={B_sp}")

    return (S_sp / S_all, B_sp / B_all)  # (f_sig, f_bkg)

def asimov_z(S: float, B: float, eps: float = 1e-12) -> float:
    """Asimov/AMS significance, numerically safe."""
    if S <= 0:
        return 0.0
    B = max(B, eps)
    return float(np.sqrt(2.0 * ((S + B) * np.log1p(S / B) - S)))

def make_xyw(
    df: pd.DataFrame,
    features: List[str],
    target_col: str = "target",
    weight_evt_col: str = "gen_weight",          # raw Event.Weight
    weight_xs_col: str = "sample_weight",    # Event.CrossSection * Event.Weight
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Missing features: {missing}")

    X = df[features].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.int32)

    # raw generator weights (Event.Weight)
    if weight_evt_col in df.columns:
        w_evt = df[weight_evt_col].to_numpy(dtype=np.float64)
    else:
        w_evt = np.ones(len(df), dtype=np.float64)

    # physics eval weight (xs * Event.Weight)
    if weight_xs_col in df.columns:
        w_xs = df[weight_xs_col].to_numpy(dtype=np.float64)
    else:
        w_xs = np.ones(len(df), dtype=np.float64)

    return X, y, w_evt, w_xs



# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
def safe_auc(y_true: np.ndarray, y_score: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    # If only one class present, roc_auc_score errors; return NaN
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score, sample_weight=sample_weight))


def weighted_significance_scan(
    y_true: np.ndarray,
    y_score: np.ndarray,
    w_xs: np.ndarray,    # 2 * lumi * evt_xs * gen_weight / sum_w_all
    *,
    lumi: float = 3000.0,
    frac_sig: float = 1.0,
    frac_bkg: float = 1.0,
    n_steps: int = 200,
    min_nS: int = 50,
    min_nB: int = 1000,
    min_B: float = 1.0,
    eps_plateau: float = 0.02,
) -> Dict[str, float]:
    """
    Scan thresholds using Asimov Z (approx).
    Uses your yield definition:
      S = lumi * sum(w_xs_sel_sig) / sum(w_evt_sig)
      B = lumi * sum(w_xs_sel_bkg) / sum(w_evt_bkg)

    """
    qs = np.linspace(0.0, 1.0, n_steps)
    thresholds = np.unique(np.quantile(y_score, qs))
    thresholds = thresholds[(thresholds > 0.0) & (thresholds < 1.0)]
    if len(thresholds) == 0:
        thresholds = np.array([0.5], dtype=float)

    thr_list, Z_list, S_list, B_list, nS_list, nB_list = [], [], [], [], [], []

    # denominators per class (avoid mixing sig/bkg)
    sig = (y_true == 1)
    bkg = (y_true == 0)

    for thr in thresholds:
        sel = (y_score >= thr)

        selS = sel & sig
        selB = sel & bkg

        S = float(w_xs[selS].sum() / lumi / frac_sig)
        B = float(w_xs[selB].sum() / lumi / frac_bkg)

        nS = lumi * S
        nB = lumi * B

        if (nS >= min_nS) and (nB >= min_nB) and (B >= min_B) and (S > 0.0):
            Z = asimov_z(nS, nB)
        else:
            Z = 0.0


        thr_list.append(float(thr))
        Z_list.append(float(Z))
        S_list.append(float(S))
        B_list.append(float(B))
        nS_list.append(nS)
        nB_list.append(nB)

    thr_arr = np.array(thr_list)
    Z_arr = np.array(Z_list)

    # robust choice: plateau threshold
    Zmax = float(Z_arr.max()) if len(Z_arr) else 0.0
    if Zmax <= 0:
        best_idx = int(np.argmax(Z_arr)) if len(Z_arr) else 0
        best_thr = float(thr_arr[best_idx]) if len(thr_arr) else 0.5
        method = "argmax_fallback"
    else:
        good = Z_arr >= (1.0 - eps_plateau) * Zmax
        best_thr = float(thr_arr[good].min())
        method = "plateau"

    # exact best at that chosen threshold
    i_star = int(np.where(thr_arr == best_thr)[0][0])

    return {
        "best_thr": best_thr,
        "best_Z": float(Z_list[i_star]),
        "best_S": float(S_list[i_star]),
        "best_B": float(B_list[i_star]),
        "best_nS": float(nS_list[i_star]),
        "best_nB": float(nB_list[i_star]),
        "method": method,
        "split_frac_sig": frac_sig,
        "split_frac_bkg": frac_bkg,
        "constraints": {"min_nS": min_nS, "min_nB": min_nB, "min_B": min_B, "eps_plateau": eps_plateau},
        "scan": {
            "thr": thr_list,
            "Z": Z_list,
            "S": S_list,
            "B": B_list,
            "nS": nS_list,
            "nB": nB_list,
        },
        # useful for debugging:
        "raw_argmax": {
            "thr": float(thr_arr[int(np.argmax(Z_arr))]),
            "Z": float(Zmax),
        },
    }



# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int = 42,
    max_depth: int = 4,
    n_estimators: int = 800,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    scale_pos_weight: float | None = None,
    tree_method: str = "hist",
) -> xgb.XGBClassifier:
    params = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        min_child_weight=min_child_weight,
        gamma=gamma,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=seed,
        tree_method=tree_method,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = float(scale_pos_weight)

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def compute_scale_pos_weight(y: np.ndarray) -> float:
    # typical choice: neg/pos
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos <= 0:
        return 1.0
    return n_neg / n_pos


# --------------------------------------
# Main
# --------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", default="ml_outputs/splits", help="Directory containing train/val/test parquet")
    ap.add_argument("--mass", type=int, default=200, help="Signal mass tag, e.g. 200 ")
    ap.add_argument("--outdir", default="ml_models", help="Base output directory")
    ap.add_argument("--features", nargs="+", default=DEFAULT_FEATURES, help="Feature columns")
    ap.add_argument("--add-features", nargs="+",default=[], help="Extra features to append")
    ap.add_argument("--drop-features", nargs="+",default=[], help="Features to drop")

    # training hyperparams
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--n-estimators", type=int, default=1200)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.8)
    ap.add_argument("--reg-lambda", type=float, default=1.0)
    ap.add_argument("--reg-alpha", type=float, default=0.0)
    ap.add_argument("--min-child-weight", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--tree-method", default="hist", choices=["hist", "approx", "auto"], help="XGBoost tree method")

    # balancing & evaluation
    ap.add_argument("--class-balance", action="store_true", help="Use scale_pos_weight = Nneg/Npos")
    ap.add_argument("--signif-steps", type=int, default=200, help="Threshold scan steps for weighted significance")
    ap.add_argument("--lumi", type=float, default=3000.0, help="Luminosity scale factor for significance/yields")
    ap.add_argument("--min-nS", type=int, default=50)
    ap.add_argument("--min-nB", type=int, default=1000)
    ap.add_argument("--min-B", type=float, default=5.0)
    ap.add_argument("--eps-plateau", type=float, default=0.02)

    args = ap.parse_args()

    splits_dir = Path(args.splits_dir)
    outdir = Path(args.outdir) / f"sig{args.mass}"
    outdir.mkdir(parents=True, exist_ok=True)

    features = args.features + args.add_features
    features = [f for f in features if f not in args.drop_features]

    all_path = splits_dir.parent / f"dataset_sig200_vs_bkg.parquet"
    train_path = splits_dir / f"train_sig{args.mass}.parquet"
    val_path   = splits_dir / f"val_sig{args.mass}.parquet"
    test_path  = splits_dir / f"test_sig{args.mass}.parquet"

    df_all = load_split(all_path)
    df_train = load_split(train_path)
    df_val   = load_split(val_path)
    df_test  = load_split(test_path)

    t_f_sig, t_f_bkg = split_fracs_by_class(df_all, df_test)
    v_f_sig, v_f_bkg = split_fracs_by_class(df_all, df_val)

    assert set(df_train.columns) == set(df_val.columns) == set(df_test.columns), \
        "Train/val/test column mismatch"

    sanity_prepared_ml(df_train, name="train")
    sanity_prepared_ml(df_val,   name="val")
    sanity_prepared_ml(df_test,  name="test")

    X_train, y_train, _, w_xs_train = make_xyw(df_train, features)
    X_val,   y_val,   _,   w_xs_val   = make_xyw(df_val, features)
    X_test,  y_test,  _,  w_xs_test  = make_xyw(df_test, features)

    spw = compute_scale_pos_weight(y_train) if args.class_balance else None

    model = train_xgb(
        X_train, y_train,
        X_val, y_val,
        seed=args.seed,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        scale_pos_weight=spw,
        tree_method=args.tree_method,
    )

    # predict scores
    p_train = model.predict_proba(X_train)[:, 1]
    p_val   = model.predict_proba(X_val)[:, 1]
    p_test  = model.predict_proba(X_test)[:, 1]

    # metrics
    metrics = {
        "mass": args.mass,
        "features": features,
        "train": {
            "auc_unweighted": safe_auc(y_train, p_train),
            "auc_weighted": safe_auc(y_train, p_train, sample_weight=w_xs_train),
        },
        "val": {
            "auc_unweighted": safe_auc(y_val, p_val),
            "auc_weighted": safe_auc(y_val, p_val, sample_weight=w_xs_val),
            "weighted_significance_scan": weighted_significance_scan(
                y_val, p_val, w_xs_val,
                lumi=args.lumi,
                frac_sig=v_f_sig,
                frac_bkg=v_f_bkg,
                n_steps=args.signif_steps,
                min_nS=args.min_nS,
                min_nB=args.min_nB,
                min_B=args.min_B,
                eps_plateau=args.eps_plateau,
            ),
        },
        "test": {
            "auc_unweighted": safe_auc(y_test, p_test),
            "auc_weighted": safe_auc(y_test, p_test, sample_weight=w_xs_test),
            "weighted_significance_scan": weighted_significance_scan(
                y_test, p_test, w_xs_test,
                lumi=args.lumi,
                frac_sig=t_f_sig,
                frac_bkg=t_f_bkg,
                n_steps=args.signif_steps,
                min_nS=args.min_nS,
                min_nB=args.min_nB,
                min_B=args.min_B,
                eps_plateau=args.eps_plateau,
            ),
        },
        "xgb": {
            "best_iteration": int(getattr(model, "best_iteration", -1)),
            "params": model.get_params(),
            "scale_pos_weight": None if spw is None else float(spw),
        },
    }

    # save model
    model_path_joblib = outdir / "model.joblib"
    print(f"Saving model to {model_path_joblib}")
    joblib.dump(model, model_path_joblib)

    model_path_ubj = outdir / "model.ubj"
    booster = model.get_booster()
    print(f"Saving native booster format to {model_path_ubj}")
    booster.save_model(model_path_ubj.as_posix())

    # save metrics
    metrics_path = outdir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # save preds for inspection
    def save_preds(df: pd.DataFrame, scores: np.ndarray, split: str):
        out = df.copy()
        out["bdt_score"] = scores.astype(np.float32)
        out_path = outdir / f"preds_{split}.parquet"
        out.to_parquet(out_path, index=False)
        return out_path

    pred_train_path = save_preds(df_train, p_train, "train")
    pred_val_path   = save_preds(df_val, p_val, "val")
    pred_test_path  = save_preds(df_test, p_test, "test")

    print(f"\nSaved model: {model_path_ubj}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved preds:   {pred_train_path}")
    print(f"               {pred_val_path}")
    print(f"               {pred_test_path}")

    # print quick headline results
    best_val = metrics["val"]["weighted_significance_scan"]
    best_test = metrics["test"]["weighted_significance_scan"]
    print("\nValidation best (weighted) Z_Asimov:   "
          f"Z={best_val['best_Z']:.4g} at thr={best_val['best_thr']:.3f} "
          f"(S={best_val['best_S']:.4g}, B={best_val['best_B']:.4g})")
    print("Test best (weighted) Z_Asimov:         "
          f"Z={best_test['best_Z']:.4g} at thr={best_test['best_thr']:.3f} "
          f"(S={best_test['best_S']:.4g}, B={best_test['best_B']:.4g})")


if __name__ == "__main__":
    main()

