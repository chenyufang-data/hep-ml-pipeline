# Model Card — sig200 BDT Classifier

## Model Overview

- **Model type:** Gradient Boosted Decision Tree (XGBoost)
- **Task:** Binary classification (signal vs background)
- **Signal:** BSM Higgs-like process at m = 200 GeV
- **Backgrounds:** bbj, ccj, bbc processes
- **Framework:** XGBoost (Python API)

This model is part of a fully reproducible HEP–ML pipeline.

---

## Intended Use

This model is intended for:
- collider phenomenology studies
- signal–background discrimination
- sensitivity projections using Z\_Asimov significance

It is **not** intended for direct deployment in an online trigger system.

---

## Training Data

- Events generated using MadGraph + Pythia + Delphes
- Features derived from reconstructed jets and kinematics
- Labels:
  - `target = 1` → signal
  - `target = 0` → background

Training is performed on **unweighted data** to avoid bias.

---

## Evaluation Data

- Independent train/val/test splits
- Physics-normalized weights applied only at evaluation time
- Full-stat projections recovered using weighted split fractions

---

## Metrics

Primary metrics are stored in:

- `ml_models/final/sig200_v1/metrics.json`
- `docs/reports/report_infer_test_sig200.latest.json`

Key metrics include:
- ROC AUC (unweighted)
- Z\_Asimov significance (weighted)
- S/√B (weighted)
- Signal and background efficiencies

For human-readable summary:

- 📄 [Human-readable report](docs/reports/report_infer_test_sig200.latest.md)

---

## Interpretability

Feature importance is evaluated using:

- SHAP summary plots
- Drop-1 feature ablation

Curated figures are available under:

- 📁 [docs/figures/](docs/figures/)

---

## Weights

- **gen_weight:** generator-level MC weight
- **sample_weight:** physics-normalized weight (includes luminosity)

Training does **not** use physics weights.  
Weights are applied **only during evaluation**.

This follows standard ML best practice: models are trained on unbiased data distributions and evaluated using domain-specific weights.

---

## Release Provenance

- **Release:** sig200_v1
- **Created:** 2026-01-06
- **Pipeline:** Makefile + freeze_final.py
- **Config:** configs/path.yaml (full), configs/path.ci.yaml (CI)
- **Code version:** (git commit hash)

---

## Limitations

- Performance depends on generator settings and detector simulation.
- Results are specific to the chosen feature set and mass point.
- Not validated on real detector data.

---

## Ethical Considerations

This model is for research and educational purposes only.

---

## References

Chenyu Fang, Wei-Shu Hou, Chung Kao and Mohamed Krabg, *Enhanced Charged Higgs Signal at the LHC*, arXiv:2511.19604 [hep-ph], 2025.  
https://doi.org/10.48550/arXiv.2511.19604

---

## Contact

Maintained by: Chenyu Fang  