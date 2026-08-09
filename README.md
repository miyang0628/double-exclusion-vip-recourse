# Double Exclusion in AI-Driven Financial Opportunity Allocation

**A reproducibility repository for auditing intersectional fairness in both VIP approval and post-rejection recourse.**

> **Anonymized submission.** This repository accompanies a manuscript under double-blind review at *[ANONYMIZED VENUE]*. All author-identifying information has been removed for review. A de-anonymized version with full attribution will be released upon acceptance.

---

## Overview

Conventional fairness audits of financial AI stop at the decision itself — *who is approved and who is rejected*. This project asks a second question that the literature has largely left open: **when a member is rejected, is the recourse advice they receive actually actionable?**

We show that an intersectionally marginalized group (older women with short credit tenure) is excluded **twice**:

1. **Axis 1 — Approval.** They are structurally far less likely to be approved for VIP status in the first place.
2. **Axis 2 — Recourse.** Even when rejected, the "how to qualify" guidance they receive is overwhelmingly *infeasible* — it asks them to change immutable attributes (age, gender, tenure) or to move actionable features far beyond a realistic action horizon.

The two axes are combined into a single **Double Exclusion Index (DEI)** that summarizes the joint probability of being both rejected *and* left without any reliable recourse path.

This repository contains the full, seed-fixed pipeline needed to reproduce every number in the manuscript.

---

## Key findings (reproducible)

| Result | Value |
|---|---|
| Model-based CDR (approval disparity, advantaged ÷ disadvantaged) | **17.2×** |
| CDR sensitivity range across 9 cutoff configurations | **10.5× – 26.0×** |
| Reliable recourse rate — advantaged (Scenario C) | **~15%** |
| Reliable recourse rate — disadvantaged (full pool, n=785) | **0.127%** (1 of 785) |
| Exact 95% upper bound — disadvantaged reliable recourse | **0.708%** |
| Double Exclusion Index — disadvantaged vs. advantaged | **0.98 vs. 0.54** |

Both axes remain directionally stable under every tested group-definition cutoff, indicating the phenomenon is not an artifact of threshold choice.

---

## Repository structure

```
double-exclusion-vip-recourse/
│
├── README.md                          # this file
├── requirements.txt                   # pinned dependencies
├── LICENSE                            # (add before public release)
│
├── notebooks/
│   ├── 01_preprocessing_grouping_model.ipynb   # data → groups → classifier
│   ├── 02_recourse_generation.ipynb            # DiCE recourse (A/B/C scenarios)
│   ├── 03_metrics_bootstrap_dei.ipynb          # metrics, bootstrap CIs, DEI
│   └── 04_sensitivity_analysis.ipynb           # cutoff robustness (both axes)
│
├── artifacts/                         # generated checkpoints (git-ignored)
│   ├── nb1_artifacts.pkl              #   model + splits + groups + CDR
│   ├── recourse_main.pkl             #   sampled recourse (n=200 per group)
│   ├── recourse_dis_full.pkl         #   full-pool recourse (disadvantaged)
│   └── nb2_frames.pkl                #   rejected / full-pool frames
│
└── data/                             # NOT included — see "Data access" below
    └── (place the 3 required source tables here)
```

### Notebook dependency graph

```
   01_preprocessing_grouping_model
                │
                │  writes nb1_artifacts.pkl
                ▼
   02_recourse_generation ──────────┐
                │                    │  reads nb1_artifacts.pkl
                │  writes            │
                │  recourse_*.pkl    │
                ▼                    ▼
   03_metrics_bootstrap_dei     04_sensitivity_analysis
   (reads nb1 + recourse_*)     (reads nb1; regenerates recourse
                                 for varied cutoffs)
```

Notebook 1 is run **once**; its artifacts are cached to disk. Notebooks 2–4 read those artifacts, so they can be re-run independently without repeating the expensive 3M-row merge and model training.

---

## Pipeline stages

**Notebook 1 — Preprocessing, grouping, model.**
Loads the three required source tables (member / approval / credit), keeps internal behavioral variables only (external credit-bureau sheets are deliberately excluded to preserve the "limited-information" assumption), binarizes the VIP target, and defines the intersectional advantaged / disadvantaged groups using **data-driven tenure cutoffs** (25th / 75th percentiles). Trains a class-balanced Random Forest and checkpoints everything.

**Notebook 2 — Recourse generation.**
Uses DiCE to generate counterfactual recourse for rejected members under three scenarios: **A** (unconstrained), **B** (immutability enforced), **C** (full guardrail: actionable features only, within a ±20% action horizon). Runs both a sampled comparison (n=200 per group) and a **full-population** pass over all disadvantaged rejected members, so the reliability bound is measured rather than extrapolated.

**Notebook 3 — Metrics, bootstrap, DEI.**
Computes per-scenario Recourse Rate, Pathway-Reliable RR, and Borrower-Reliable RR; attaches percentile bootstrap 95% CIs and exact (Clopper–Pearson) bounds for zero-count cells; and combines both axes into the Double Exclusion Index.

**Notebook 4 — Sensitivity analysis.**
Sweeps the group-definition cutoffs (3 tenure percentile pairs × 3 age thresholds) and confirms both the approval disparity (Axis 1) and the recourse-reliability gap (Axis 2) persist across all configurations. Includes an inline guard that prevents a silent misconfiguration of the counterfactual engine.

---

## Reproducibility notes

- **Fixed seeds throughout** (`random_state=42`; bootstrap seed fixed). Reported figures are stable across re-runs.
- **Unified counterfactual configuration.** All twelve features are declared continuous for the recourse engine so that the main analysis (Notebook 2/3) and the sensitivity analysis (Notebook 4) come from one identical configuration. A runtime assertion enforces this.
- **Feature governance is explicit.** The immutable / actionable / non-controllable partition is defined once and drives *both* the model features and the recourse penalties, so it is fully auditable.
- **Measured, not extrapolated.** The headline disadvantaged reliability bound is computed over the entire rejected pool (n=785), not inferred from a small sample.

---

## Environment

```bash
python -m venv venv && source venv/bin/activate      # or conda
pip install -r requirements.txt
```

Core dependencies (see `requirements.txt` for pinned versions):
`pandas`, `numpy`, `scikit-learn`, `dice-ml`, `scipy`, `jupyter`.

---

## Data access

The analysis uses a public synthetic financial (card-usage) dataset. The raw files are **not redistributed** in this repository. To reproduce:

1. Obtain the three required source tables (member information, credit information, approval/sales) from the original public provider.
2. Place them under `data/` (or set the `DATA_DIR` path at the top of each notebook).
3. Run the notebooks in order (1 → 2 → 3 → 4).

> **Note.** Only member / credit / approval tables are needed. External credit-bureau tables are intentionally not used.

---

## How to run

```
# 1. one-time: build model + groups (writes artifacts/)
notebooks/01_preprocessing_grouping_model.ipynb

# 2. generate recourse (longest stage; checkpoints as it goes)
notebooks/02_recourse_generation.ipynb

# 3. compute all metrics + DEI (fast; re-runnable)
notebooks/03_metrics_bootstrap_dei.ipynb

# 4. cutoff robustness for both axes (fast–moderate)
notebooks/04_sensitivity_analysis.ipynb
```

Each notebook has a single `DATA_DIR` variable at the top to set your local path.

---

## Citation

```
Anonymous Author(s). "Double Exclusion: Auditing Intersectional Fairness
in AI-Driven Financial Opportunity Allocation and Recourse."
Manuscript under review at [ANONYMIZED VENUE], 2026.
```

*(Full citation and author information will be added upon acceptance.)*

---

## License

To be added before public (de-anonymized) release.
