#!/usr/bin/env python3
"""
summarize_inference.py

Summarize inference results produced by a frozen, versioned model.
Computes aggregate metrics and optional per-sample breakdowns from
the inference outputs.

Inputs (default):
  ml_models/final/sig{MASS}/preds_infer_test.parquet
  ml_models/final/sig{MASS}/threshold.json

Optional:
  ml_models/final/sig{MASS}/preds_infer_test.meta.json

Outputs:
  ml_models/final/sig{MASS}/report_infer_test.json
  ml_models/final/sig{MASS}/report_by_sample.csv (optional) 
  doc/reports/report_infer_test_sig{MASS}.latest.json
  doc/reports/report_infer_test_sig{MASS}.latest.md
  doc/figures/*_sig{MASS}.png

Example:
  python src/summarize_inference.py --mass 200 \
    --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional
import re
from datetime import datetime
import shutil

import numpy as np
import pandas as pd

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

def z_asimov(S: float, B: float) -> float:
    """Asimov significance (approx)."""
    if S <= 0.0 or B <= 0.0:
        return 0.0
    return float(np.sqrt(2.0 * ((S + B) * np.log(1.0 + S / B) - S)))


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def write_industry_md_report(report: Dict[str, Any], out_path: Path):
    """
    Write a concise markdown summary from the report dict.
    """
    mass = report.get("signal_mass", "NA")
    split = report.get("split", "NA")
    n_rows = report.get("n_rows", "NA")
    thr = report.get("threshold", "NA")

    score_stats = report.get("score_stats", {})
    yields = report.get("yields_weighted", {})
    by_sample = report.get("by_sample", {}).get("top_by_weight_sum", [])

    Z_full = yields.get("Z_Asimov_pass_full", None)
    Z_eval = yields.get("Z_Asimov_pass_eval", None)
    S_eff = yields.get("S_eff", None)
    B_eff = yields.get("B_eff", None)
    S_pass_full = yields.get("S_pass_full", None)
    B_pass_full = yields.get("B_pass_full", None)

    date_str = datetime.now().strftime("%Y-%m-%d")

    lines = []

    # Header
    lines.append(f"# Model Inference Summary — sig{mass} ({split})")
    lines.append(f"> Generated on {date_str} by `summarize_inference.py`")
    lines.append("")

    # Executive summary
    lines.append("## Executive Summary")
    if Z_full is not None:
        lines.append(
            f"- The model achieves **Z\_Asimov = {Z_full:.2f} σ** at the selected threshold, "
            f"with **{S_eff*100:.1f}% signal efficiency** and **{B_eff*100:.2f}% background efficiency**."
        )
    else:
        lines.append("- Inference completed successfully. See metrics below for details.")
    lines.append("")

    # Dataset
    lines.append("## Dataset")
    lines.append(f"- **Signal mass:** {mass} GeV")
    lines.append(f"- **Split:** {split}")
    lines.append(f"- **Rows evaluated:** {n_rows}")
    lines.append("")

    # Model decision
    lines.append("## Decision Threshold")
    if isinstance(thr, float):
        lines.append(f"- **BDT threshold:** {thr:.3f}")
    else:
        lines.append(f"- **BDT threshold:** {thr}")
    lines.append("")

    # Core metrics
    lines.append("## Key Metrics")
    if Z_full is not None:
        lines.append(f"- **Z\_Asimov (full-stat, pass):** {Z_full:.3f} σ")
    if Z_eval is not None:
        lines.append(f"- **Z\_Asimov (evaluated split):** {Z_eval:.3f} σ")
    if S_eff is not None:
        lines.append(f"- **Signal efficiency:** {S_eff*100:.2f}%")
    if B_eff is not None:
        lines.append(f"- **Background efficiency:** {B_eff*100:.2f}%")
    if S_pass_full is not None and B_pass_full is not None:
        lines.append(f"- **Expected S (pass):** {S_pass_full:,.1f} events")
        lines.append(f"- **Expected B (pass):** {B_pass_full:,.1f} events")
    lines.append("")

    # Score distribution
    lines.append("## Score Distribution")
    if score_stats:
        lines.append(
            f"- **Min / Mean / Max:** "
            f"{score_stats.get('min', float('nan')):.3f} / "
            f"{score_stats.get('mean', float('nan')):.3f} / "
            f"{score_stats.get('max', float('nan')):.3f}"
        )
        lines.append(
            f"- **P50 / P90 / P95 / P99:** "
            f"{score_stats.get('p50', float('nan')):.3f} / "
            f"{score_stats.get('p90', float('nan')):.3f} / "
            f"{score_stats.get('p95', float('nan')):.3f} / "
            f"{score_stats.get('p99', float('nan')):.3f}"
        )
    lines.append("")

    # Top contributors
    if by_sample:
        lines.append("## Top Contributors (by weight)")
        lines.append("| Sample | Events | Pass rate | Weight sum | Weight pass |")
        lines.append("|--------|--------|-----------|------------|-------------|")
        for row in by_sample:
            lines.append(
                f"| {row.get('sample','')} | {row.get('n','')} | "
                f"{row.get('pass_rate',0)*100:.1f}% | "
                f"{row.get('w_sum',0):.1f} | {row.get('w_pass',0):.1f} |"
            )
        lines.append("")

    # Notes
    lines.append("## Notes")
    lines.append("- `sample_weight` includes luminosity.")
    lines.append("- Full-stat metrics are scaled using split metadata (`weighted_frac_*`).")
    lines.append("- This summary is intended for quick inspection; see the JSON report for full details.")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"[OK] Wrote markdown report: {out_path}")

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
    ap.add_argument("--split", default="test", choices=["train", "val", "test"], help="Which infer file to summarize")
    ap.add_argument("--preds-name", default=None,
                    help="Override preds filename (default: preds_infer_{split}.parquet)")
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet", help="Preds format")
    ap.add_argument("--out-name", default=None,
                    help="Override report filename (default: report_infer_{split}.json)")
    ap.add_argument("--write-by-sample-csv", action="store_true",
                    help="Write per-sample summary CSV next to JSON report")
    ap.add_argument("--lumi", type=float, default=3000, help="Luminosity for Z_Asimov evaluation, e.g. 3000")
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

    preds_name = args.preds_name or f"preds_infer_{args.split}.{args.format}"
    meta_name  = Path(preds_name).with_suffix(".meta.json")
    out_name   = args.out_name or f"report_infer_{args.split}.json"

    preds_path = model_dir / preds_name
    thr_path   = model_dir / "threshold.json"
    meta_path  = model_dir / meta_name
    out_path   = model_dir / out_name

    if not preds_path.exists():
        raise FileNotFoundError(f"Missing preds file: {preds_path}")
    if not thr_path.exists():
        raise FileNotFoundError(f"Missing threshold.json: {thr_path}")

    # load threshold
    thr_json = json.loads(thr_path.read_text())
    thr = float(thr_json["threshold"])

    # load preds 
    df = pd.read_parquet(preds_path)

    required = ["bdt_score"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Required column '{c}' not found in {preds_path}")

    # if bdt_pred missing, derive from threshold
    if "bdt_pred" not in df.columns:
        df["bdt_pred"] = (df["bdt_score"] >= thr).astype(np.int8)

    # optional columns
    has_target = "target" in df.columns
    has_weight = "sample_weight" in df.columns
    has_sample = "sample" in df.columns

    # basic stats
    n_rows = int(len(df))
    score = df["bdt_score"].to_numpy(dtype=np.float64)
    pred = df["bdt_pred"].to_numpy(dtype=np.int8)

    report: Dict[str, Any] = {
        "signal_mass": str(args.mass),
        "model_dir": str(model_dir),
        "split": args.split,
        "preds_file": preds_name,
        "n_rows": n_rows,
        "threshold": thr,
        "threshold_source": "threshold.json",
        "score_stats": {
            "min": safe_float(np.min(score)) if n_rows else float("nan"),
            "mean": safe_float(np.mean(score)) if n_rows else float("nan"),
            "max": safe_float(np.max(score)) if n_rows else float("nan"),
            "p50": safe_float(np.quantile(score, 0.50)) if n_rows else float("nan"),
            "p90": safe_float(np.quantile(score, 0.90)) if n_rows else float("nan"),
            "p95": safe_float(np.quantile(score, 0.95)) if n_rows else float("nan"),
            "p99": safe_float(np.quantile(score, 0.99)) if n_rows else float("nan"),
        },
        "pass_rate_unweighted": safe_float(pred.mean()) if n_rows else 0.0,
        "columns_present": {
            "target": bool(has_target),
            "sample": bool(has_sample),
            "sample_weight": bool(has_weight),
            "bdt_pred": True,
        },
        "provenance": {},
    }

    # attach optional meta provenance 
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            report["provenance"]["preds_meta"] = meta
        except Exception as e:
            report["provenance"]["preds_meta_error"] = str(e)

    # confusion matrix + weighted yields
    if has_target:
        y = df["target"].to_numpy(dtype=np.int8)
        tp = int(((y == 1) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        tn = int(((y == 0) & (pred == 0)).sum())

        report["confusion_unweighted"] = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
        report["rates_unweighted"] = {
            "tpr": safe_float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan"),
            "fpr": safe_float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan"),
            "tnr": safe_float(tn / (fp + tn)) if (fp + tn) > 0 else float("nan"),
            "fnr": safe_float(fn / (tp + fn)) if (tp + fn) > 0 else float("nan"),
        }

        if has_weight:
            # preds meta may or may not exist
            meta = report.get("provenance", {}).get("preds_meta", {}) if "provenance" in report else {}
            wf = None
            if isinstance(meta, dict) and meta.get("input_is_split") and isinstance(meta.get("weighted_frac_current_split"), dict):
                wf = meta["weighted_frac_current_split"]

            f_sig = float(wf.get("f_sig", 1.0)) if wf else 1.0
            f_bkg = float(wf.get("f_bkg", 1.0)) if wf else 1.0

            wxs = df["sample_weight"].to_numpy(dtype=np.float64)

            S_pass_eval = float(wxs[(y==1) & (pred==1)].sum())
            B_pass_eval = float(wxs[(y==0) & (pred==1)].sum())
            S_all_eval  = float(wxs[(y==1)].sum())
            B_all_eval  = float(wxs[(y==0)].sum())

            # optional scale-back
            S_pass_full, B_pass_full = S_pass_eval, B_pass_eval
            S_all_full, B_all_full = S_all_eval, B_all_eval

            if wf is not None:
                f_sig = float(wf.get("f_sig", 1.0))
                f_bkg = float(wf.get("f_bkg", 1.0))
                if f_sig > 0: 
                    S_pass_full = S_pass_eval / f_sig
                    S_all_full  = S_all_eval  / f_sig
                if f_bkg > 0:
                    B_pass_full = B_pass_eval / f_bkg
                    B_all_full  = B_all_eval  / f_bkg
            
            report["yields_weighted"] = {
                "sample_weight_includes_lumi": True,
                "weighted_frac_used": wf,

                # what this file contains (split-level)
                "S_pass_eval": S_pass_eval,
                "B_pass_eval": B_pass_eval,
                "S_all_eval":  S_all_eval,
                "B_all_eval":  B_all_eval,
                "Z_Asimov_pass_eval": z_asimov(S_pass_eval, B_pass_eval),
                "S_over_sqrtB_pass_eval": safe_float(S_pass_eval / np.sqrt(B_pass_eval)) if B_pass_eval > 0 else 0.0,

                # scaled-back (HEP “full-stat equivalent”)
                "S_pass_full": S_pass_full,
                "B_pass_full": B_pass_full,
                "S_all_full":  S_all_full,
                "B_all_full":  B_all_full,
                "Z_Asimov_pass_full": z_asimov(S_pass_full, B_pass_full),
                "S_over_sqrtB_pass_full": safe_float(S_pass_full / np.sqrt(B_pass_full)) if B_pass_full > 0 else 0.0,
                "S_eff": safe_float(S_pass_full / S_all_full) if S_all_full > 0 else float("nan"),
                "B_eff": safe_float(B_pass_full / B_all_full) if B_all_full > 0 else float("nan"),
            }

    # by-sample breakdown 
    if has_sample:
        group_cols = ["sample"]
        agg: Dict[str, Any] = {"n": ("bdt_score", "size"),
                              "pass_rate": ("bdt_pred", "mean"),
                              "score_mean": ("bdt_score", "mean"),
                              "score_p95": ("bdt_score", lambda s: float(np.quantile(s, 0.95)))}
        if has_weight:
            agg["w_sum"] = ("sample_weight", "sum")
            agg["w_pass"] = ("sample_weight", lambda s: float(s[df.loc[s.index, "bdt_pred"] == 1].sum()))

        by = (df.groupby(group_cols, sort=False)
                .agg(**agg)
                .reset_index())

        # clean types
        by["pass_rate"] = by["pass_rate"].astype(float)

        report["by_sample"] = {
            "n_samples": int(len(by)),
            "top_by_weight_sum": by.sort_values("w_sum", ascending=False).head(10).to_dict(orient="records")
            if has_weight and "w_sum" in by.columns else
            by.sort_values("n", ascending=False).head(10).to_dict(orient="records"),
        }

        if args.write_by_sample_csv:
            csv_path = model_dir / f"report_by_sample_{args.split}.csv"
            by.to_csv(csv_path, index=False)
            report["by_sample_csv"] = str(csv_path)

    # write report
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[OK] Wrote report: {out_path}")

    # quick console summary
    print(f"[INFO] n_rows={n_rows}, pass_rate={report['pass_rate_unweighted']:.4f}, thr={thr:.6f}")
    if "yields_weighted" in report:
        yw = report["yields_weighted"]
        print(f"[INFO] Weighted pass: S={yw['S_pass_eval']:.4e}, B={yw['B_pass_eval']:.4e}, Z_Asimov={yw['Z_Asimov_pass_eval']:.4g}")

    # copy summary and figures to stable directory
    docs_fig = Path("docs/figures")
    docs_fig.mkdir(parents=True, exist_ok=True)

    mapping = {
        "shap_summary.png": "shap_sig200.png",
        "importance_permutation.png": "importance_perm_sig200.png",
        "scores_train_vs_test.png": "scores_train_vs_test_sig200.png",
        "roc_val.png": "roc_val_sig200.png",
        "roc_test.png": "roc_test_sig200.png",
        "Zscan_val.png": "zscan_val_sig200.png",
        "Zscan_test.png": "zscan_test_sig200.png",
    }

    for src_name, dst_name in mapping.items():
        src = model_dir / "plots" / src_name
        if src.exists():
            shutil.copy2(src, docs_fig / dst_name)
            print(f"[OK] Copied figure: {src_name} -> docs/figures/{dst_name}")
        else:
            print(f"[WARN] Figure not found (skipped): {src_name}")

    docs_reports = Path("docs/reports")
    docs_reports.mkdir(parents=True, exist_ok=True)

    latest_report = docs_reports / f"report_infer_{args.split}_sig{args.mass}.latest.json"
    shutil.copy2(out_path, latest_report)
    print(f"[OK] Copied latest report: {latest_report}")

    md_path = docs_reports / f"report_infer_{args.split}_sig{args.mass}.latest.md"
    write_industry_md_report(report, md_path)

if __name__ == "__main__":
    main()

