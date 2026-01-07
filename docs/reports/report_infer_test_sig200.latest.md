# Model Inference Summary — sig200 (test)
> Generated on 2026-01-06 by `summarize_inference.py`

## Executive Summary
- The model achieves **Z\_Asimov = 6.17 σ** at the selected threshold, with **41.2% signal efficiency** and **7.66% background efficiency**.

## Dataset
- **Signal mass:** 200 GeV
- **Split:** test
- **Rows evaluated:** 4047

## Decision Threshold
- **BDT threshold:** 0.573

## Key Metrics
- **Z\_Asimov (full-stat, pass):** 6.174 σ
- **Z\_Asimov (evaluated split):** 1.911 σ
- **Signal efficiency:** 41.16%
- **Background efficiency:** 7.66%
- **Expected S (pass):** 2,581.7 events
- **Expected B (pass):** 173,986.2 events

## Score Distribution
- **Min / Mean / Max:** 0.010 / 0.315 / 0.910
- **P50 / P90 / P95 / P99:** 0.254 / 0.696 / 0.772 / 0.828

## Top Contributors (by weight)
| Sample | Events | Pass rate | Weight sum | Weight pass |
|--------|--------|-----------|------------|-------------|
| bkg_bbj | 869 | 7.6% | 138868.6 | 10412.6 |
| bkg_ccj | 607 | 10.5% | 52806.3 | 5280.2 |
| bkg_bbc | 1271 | 5.6% | 39668.2 | 2037.8 |
| sig_200 | 1300 | 41.5% | 619.6 | 255.0 |

## Notes
- `sample_weight` includes luminosity.
- Full-stat metrics are scaled using split metadata (`weighted_frac_*`).
- This summary is intended for quick inspection; see the JSON report for full details.

