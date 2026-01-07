# Weights and Significance Metrics

This page explains the meaning and usage of `gen_weight`, `sample_weight`, and the Z\_Asimov significance metric in this repository.

The goal is to clearly separate:
- **simulation bookkeeping**
- **physics normalization**
- **ML training vs evaluation logic**

---

## 1. `gen_weight`

**Definition:**  
`gen_weight` is the event-level weight from the Monte Carlo generator (e.g. MadGraph/Delphes).

It typically encodes:
- generator cross section
- filter efficiency
- event reweighting factors

**Purpose:**  
Used to normalize raw events to physical cross sections.

**Important:**  
`gen_weight` is **not** used directly for ML training. It is only used to build `sample_weight`.

---

## 2. `sample_weight`

**Definition:**  
`sample_weight` is the **physics-normalized event weight** used for evaluation.

In this pipeline, it is constructed as:

sample_weight = (cross section × luminosity) / N_generated × gen_weight


So **`sample_weight` already includes luminosity**.

**Purpose:**
- compute expected event yields
- compute Z\_Asimov
- compute S/B, efficiencies, pass rates

**Usage:**
- ❌ **Not used for ML training** (training is unweighted or class-balanced)
- ✅ **Used for evaluation and reporting only**

> This follows standard ML best practice: models are trained on unbiased data distributions and evaluated using domain-specific weights.

---

## 3. Why we do NOT train with `sample_weight`

Using physics weights during training can:
- bias the classifier toward high-weight background
- reduce generalization
- hurt stability

Therefore:
- Training uses **unweighted events** (or optional class balance)
- Physics meaning is injected **only at evaluation time**

This clean separation is intentional.

---

## 4. Z\_Asimov significance

We use the standard Asimov approximation:

Z = sqrt( 2 * [ (S + B) * ln(1 + S / B) - S ] )

Where:
- **S** = expected signal yield after selection
- **B** = expected background yield after selection

Both S and B are computed using **`sample_weight`**.

**Interpretation:**
- Z ≈ 3 → evidence
- Z ≈ 5 → discovery-level sensitivity (HEP convention)

---

## 5. Full-stat vs split-stat scaling

During training we use train/val/test splits.  
To recover full-dataset expectations, we scale using:

weighted_frac_train / weighted_frac_val / weighted_frac_test

stored in:

ml_outputs/splits/split_sig{MASS}.meta.json


This allows:

- honest split-based ML evaluation
- correct full-statistics physics projections

---

## 6. Summary

| Quantity        | Used for training | Used for evaluation | Contains luminosity |
|-----------------|-------------------|---------------------|---------------------|
gen_weight        | ❌                | ⚠️ indirect        | ❌                  |
sample_weight     | ❌                | ✅                 | ✅                  |
Z\_Asimov         | ❌                | ✅                 | ✅                  |

---

## 7. Design principle

This pipeline follows a strict separation:

> **ML optimization ≠ Physics normalization**

This makes the code:
- easier to reason about
- easier to test
- closer to industry ML workflows
- still fully correct for HEP analysis
