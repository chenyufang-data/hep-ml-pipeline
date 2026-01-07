#!/usr/bin/env python3
"""
plot_bdt_diagnostics.py

Generate standard diagnostic plots for a trained BDT model, including:
  - score distributions (train/val/test, split by class)
  - KS test for overtraining check
  - ROC curves (unweighted and weighted)
  - weighted significance scan (Z vs threshold)
  - permutation importance on the test set
  - SHAP summary plot

Expected inputs (from train_bdt.py):
  ml_models/sig{mass}/model.ubj
  ml_models/sig{mass}/preds_train.parquet
  ml_models/sig{mass}/preds_val.parquet
  ml_models/sig{mass}/preds_test.parquet

Each preds_*.parquet is expected to contain:
  - target (0/1)
  - bdt_score (0..1)
  - sample_weight
  - feature columns (required for permutation importance and SHAP)

Outputs:
  Diagnostic plots are written to:
    ml_models/sig{mass}/plots/

Example:
  python src/plot_bdt_diagnostics.py --mass 200 --modeldir ml_models
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.inspection import permutation_importance

import xgboost as xgb
import shap


# ----------------------------
# Helpers
# ----------------------------
def _read_preds(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"target", "bdt_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    if "sample_weight" not in df.columns:
        df["sample_weight"] = 1.0
    return df

def _z_asimov(S: float, B: float) -> float:
    """Asimov significance."""
    if S <= 0.0 or B <= 0.0:
        return 0.0
    return float(np.sqrt(2.0 * ((S + B) * np.log(1.0 + S / B) - S)))

def _plot_score_hists(train_df, test_df, outpath: Path):
    """
    Score distributions train vs test, signal vs background, with KS test.
    """
    fig = plt.figure(figsize=(7, 5))

    # Split
    tr_s = train_df[train_df["target"] == 1]["bdt_score"].to_numpy()
    tr_b = train_df[train_df["target"] == 0]["bdt_score"].to_numpy()
    te_s = test_df[test_df["target"] == 1]["bdt_score"].to_numpy()
    te_b = test_df[test_df["target"] == 0]["bdt_score"].to_numpy()

    # KS tests (unweighted)
    ks_s = ks_2samp(tr_s, te_s)
    ks_b = ks_2samp(tr_b, te_b)

    bins = np.linspace(0, 1, 41)
    plt.hist(tr_s, bins=bins, density=True, histtype="step", linewidth=2, label=f"Train S (KS p={ks_s.pvalue:.3g})")
    plt.hist(tr_b, bins=bins, density=True, histtype="step", linewidth=2, label=f"Train B (KS p={ks_b.pvalue:.3g})")

    plt.hist(te_s, bins=bins, density=True, alpha=0.35, label="Test S (filled)")
    plt.hist(te_b, bins=bins, density=True, alpha=0.35, label="Test B (filled)")

    plt.xlabel("BDT score")
    plt.ylabel("Density")
    plt.title("Score distributions (overtraining check)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def _plot_roc(df, outpath: Path, title: str):
    fig = plt.figure(figsize=(6, 5))

    y = df["target"].to_numpy(dtype=int)
    s = df["bdt_score"].to_numpy(dtype=float)
    w = df["sample_weight"].to_numpy(dtype=np.float64)

    # Unweighted ROC
    fpr, tpr, _ = roc_curve(y, s)
    auc_u = roc_auc_score(y, s)

    # Weighted ROC (sklearn supports sample_weight)
    fprw, tprw, _ = roc_curve(y, s, sample_weight=w)
    auc_w = roc_auc_score(y, s, sample_weight=w)

    plt.plot(fpr, tpr, label=f"Unweighted AUC = {auc_u:.3f}")
    plt.plot(fprw, tprw, label=f"Weighted AUC = {auc_w:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def _plot_significance(
    model_dir: Path,
    outpath: Path,
    title: str,
    split: str,
):
    """
    Plot weighted Z_Asimov vs threshold from metrics.json.
    """
    fig = plt.figure(figsize=(7, 5))

    model_path = model_dir / "metrics.json"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} does not exist")

    with open(model_path) as f:
        metrics = json.load(f)

    scan = metrics[split]["weighted_significance_scan"]["scan"]
    best = metrics[split]["weighted_significance_scan"]

    thr_all = np.asarray(scan["thr"], dtype=float)
    Z_all   = np.asarray(scan["Z"],  dtype=float)
    nS_all  = np.asarray(scan["nS"], dtype=float)
    nB_all  = np.asarray(scan["nB"], dtype=float)
    B_all   = np.asarray(scan["B"],  dtype=float)
    
    min_nS = best["constraints"]["min_nS"]
    min_nB = best["constraints"]["min_nB"]
    min_B = best["constraints"]["min_B"]

    mask = (nS_all >= min_nS) & (nB_all >= min_nB) & (B_all >= min_B)

    thr = thr_all[mask]
    Zs  = Z_all[mask]

    if thr.size == 0:
        raise ValueError(
            f"No scan points pass constraints."
        )

    plateau = thr[Zs >= 0.98 * Zs.max()]
    print("plateau:", plateau.min(), plateau.max())

    plt.plot(thr, Zs)
    plt.axvline(
        best["best_thr"],
        linestyle="--",
        label=(f"thr={best['best_thr']:.3f}, Z={best['best_Z']:.1f} "
               f"(nS={best['best_nS']:.3f}, nB={best['best_nB']:.1f})")
    )

    plt.xlabel("Score threshold")
    plt.ylabel(r"Weighted $Z_\mathrm{Asimov}=\sqrt{2\left((S+B)\ln(1+\frac{S}{B})-S\right)}$")
    plt.title(title + f"\n(S={best['best_S']:.3g}, B={best['best_B']:.3g})")
    plt.legend()
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

    return best, scan


def _plot_xgb_importance(model_path: Path, feature_names: list[str], outdir: Path):
    booster = xgb.Booster()
    booster.load_model(str(model_path))

    # XGBoost importance types: "gain", "weight", "cover", "total_gain", "total_cover"
    for imp_type in ["gain", "weight"]:
        score_dict = booster.get_score(importance_type=imp_type)

        # Ensure all features appear (missing => 0)
        vals = np.array(
            [score_dict.get(f"f{i}", 0.0) for i in range(len(feature_names))],
            dtype=float,
        )

        # sort
        order = np.argsort(vals)
        fig = plt.figure(figsize=(7, 5))
        plt.barh(np.array(feature_names)[order], vals[order])
        plt.xlabel(f"Importance ({imp_type})")
        plt.title(f"XGBoost feature importance ({imp_type})")
        plt.tight_layout()
        fig.savefig(outdir / f"importance_{imp_type}.png")
        plt.close(fig)


def _plot_permutation_importance(test_df: pd.DataFrame, feature_names: list[str], outdir: Path, n_repeats=10, seed=42):
    """
    Permutation importance is often more interpretable than built-in gain/weight.
    Needs features present in preds_test.parquet.
    """
    missing = [f for f in feature_names if f not in test_df.columns]
    if missing:
        print(f"[perm importance] Skip: missing features in test preds parquet: {missing}")
        return

    X = test_df[feature_names].to_numpy(dtype=np.float32)
    y = test_df["target"].to_numpy(dtype=np.int32)
    w = test_df["sample_weight"].to_numpy(dtype=np.float64)

    # Wrap a saved XGBoost Booster as a sklearn-compatible estimator
    # so it can be used with sklearn utilities.
    from sklearn.base import BaseEstimator, ClassifierMixin

    class BoosterWrapper(BaseEstimator, ClassifierMixin):
        def __init__(self, model_path: str, feature_names: list[str]):
            self.model_path = model_path
            self.feature_names = feature_names
            self.booster_ = None

        def fit(self, X, y=None):
            booster = xgb.Booster()
            booster.load_model(self.model_path)
            self.booster_ = booster
            return self

        def predict_proba(self, X):
            dm = xgb.DMatrix(X, feature_names=self.feature_names)
            p = self.booster_.predict(dm)  # shape (n,)
            p = p.reshape(-1, 1)
            return np.hstack([1 - p, p])
        
        def predict(self, X):
            return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    # load model.ubj
    model_path = outdir.parent / "model.ubj"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} does not exist")

    est = BoosterWrapper(str(model_path), feature_names).fit(X, y)

    def auc_scorer(estimator, X_, y_):
        s_ = estimator.predict_proba(X_)[:, 1]
        return roc_auc_score(y_, s_, sample_weight=w)

    result = permutation_importance(
        est,
        X,
        y,
        n_repeats=n_repeats,
        random_state=seed,
        scoring=auc_scorer,
    )

    imp = result.importances_mean
    err = result.importances_std
    order = np.argsort(imp)

    fig = plt.figure(figsize=(7, 5))
    plt.barh(np.array(feature_names)[order], imp[order], xerr=err[order])
    plt.xlabel("Permutation importance (#Delta weighted AUC)")
    plt.title("Permutation importance (test)")
    plt.tight_layout()
    fig.savefig(outdir / "importance_permutation.png")
    plt.close(fig)

def _plot_shap_summary(model_path: Path, df: pd.DataFrame, feature_names: list[str], outpath: Path, max_points=20000, seed=42):
    if shap is None:
        print("[shap] Skip: shap not installed. Install with: pip install shap")
        return

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        print(f"[shap] Skip: missing features in preds parquet: {missing}")
        return

    # load booster
    booster = xgb.Booster()
    booster.load_model(str(model_path))

    # sample rows to keep runtime/memory reasonable
    rng = np.random.default_rng(seed)
    df_ = df.sample(n=min(len(df), max_points), random_state=seed) if len(df) > max_points else df
    X = df_[feature_names].to_numpy(dtype=np.float32)

    # compute SHAP values
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X)

    # save plot 
    plt.figure(figsize=(7, 5))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        show=False,
        plot_type="dot",
    )
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

# --------------------------------------
# Main
# --------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", type=int, default=200, help="e.g. 200")
    ap.add_argument("--modeldir", default="ml_models", help="Base directory where sig{mass}/ exists")
    ap.add_argument("--features", nargs="+", default=None, help="Override feature list")
    args = ap.parse_args()

    mdir = Path(args.modeldir) / f"sig{args.mass}"
    if not mdir.exists():
        raise FileNotFoundError(f"Missing directory: {mdir}")

    # load feature list from metrics.json if not given
    metrics_path = mdir / "metrics.json"
    if args.features is None:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        feature_names = metrics["features"]
    else:
        feature_names = args.features

    train_df = _read_preds(mdir / "preds_train.parquet")
    val_df   = _read_preds(mdir / "preds_val.parquet")
    test_df  = _read_preds(mdir / "preds_test.parquet")

    outdir = mdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    _plot_score_hists(train_df, test_df, outdir / "scores_train_vs_test.png")
    _plot_roc(val_df,  outdir / "roc_val.png",  "ROC (validation)")
    _plot_roc(test_df, outdir / "roc_test.png", "ROC (test)")

    best_val, _ = _plot_significance(mdir, outdir / "Zscan_val.png", "Weighted significance scan (val)", "val")
    best_tst, _ = _plot_significance(mdir, outdir / "Zscan_test.png", "Weighted significance scan (test)", "test")

    # model feature importance
    model_path = mdir / "model.ubj"
    #_plot_xgb_importance(model_path, feature_names, outdir)

    try:
        _plot_permutation_importance(test_df, feature_names, outdir)
    except Exception as e:
        print(f"[permutation] Skip due to error: {e}")
        
    try:
        _plot_shap_summary(model_path, test_df, feature_names, outdir / "shap_summary.png")
    except Exception as e:
        print(f"[shap] Skip due to error: {e}")

    # print summary
    print(f"[plots] Saved to: {outdir}")
    print(f"[val]  best thr={best_val['best_thr']:.3f}  Z={best_val['best_Z']:.4f}  S={best_val['best_S']:.3f}  B={best_val['best_B']:.3g}")
    print(f"[test] best thr={best_tst['best_thr']:.3f}  Z={best_tst['best_Z']:.4f}  S={best_tst['best_S']:.3f}  B={best_tst['best_B']:.3g}")


if __name__ == "__main__":
    main()
