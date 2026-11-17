# Double Exclusion: Auditing Intersectional Fairness in VIP Approval and Post-Rejection Recourse

**A reproducibility repository for a manuscript under double-blind review.**

> **Anonymized submission.** This repository accompanies a manuscript submitted to an anonymized venue. All author-identifying information has been removed for review. A de-anonymized version with full attribution will be released upon acceptance.

---

## Table of contents

1. [Overview](#1-overview)
2. [Key findings](#2-key-findings-reproducible)
3. [Repository structure](#3-repository-structure)
4. [Method in brief](#4-method-in-brief)
5. [The pipeline, notebook by notebook](#5-the-pipeline-notebook-by-notebook)
6. [Result tables](#6-result-tables)
7. [Figures](#7-figures)
8. [How to run](#8-how-to-run)
9. [Configuration and reproducibility](#9-configuration-and-reproducibility)
10. [Data access](#10-data-access)
11. [Environment](#11-environment)
12. [Troubleshooting](#12-troubleshooting)
13. [License and citation](#13-license-and-citation)

---

## 1. Overview

Conventional fairness audits of financial AI stop at the decision itself — *who is approved and who is rejected*. This project asks a second question the literature has largely left open: **when a member is rejected, is the recourse advice they receive actually actionable?**

Using a large public synthetic dataset of card-usage behavior, we audit an opportunity-allocation model — selection into a premium (VIP) membership tier — along two axes and combine them into a single index:

- **Axis 1 — Approval.** A **Conditional Disparity Ratio (CDR)** between an advantaged and a disadvantaged intersectional profile.
- **Axis 2 — Recourse.** Counterfactual recourse is generated under three governance scenarios and judged **reliable** only if it (i) changes no immutable feature, (ii) keeps every actionable change within a ±20% action horizon, and (iii) leaves the issuer-controlled (non-controllable) features unchanged.
- **Double Exclusion Index (DEI).** The two axes are combined as an *exact joint probability* (chain rule) of being both rejected and left without any reliable recourse.

The disadvantaged group (older women with short credit tenure) is far less likely to be approved and, once rejected, receives virtually no feasible recourse; the recourse gap exceeds the approval gap and falls on the same members.

This repository contains the full, seed-fixed pipeline needed to reproduce every number, table, and figure in the manuscript.

---

## 2. Key findings (reproducible)

| Result | Value | Source table |
|---|---|---|
| Model performance (F1 / recall / AUC-ROC) | 0.664 / 0.572 / 0.892 | `table0_model_metrics.csv` |
| Approval disparity, baseline (CDR = advantaged ÷ disadvantaged) | **17.3×** | `table6_cdr_sensitivity.csv` |
| CDR range across 9 cutoff configurations | **10.5× – 26.0×** | `table6_cdr_sensitivity.csv` |
| First-axis gap: three-way interaction (log-odds) | ≈ 0 (0.004) | `table4b_interaction_decomposition.csv` |
| Reliable recourse — advantaged (Scenario C, n=200) | **15.0%** (95% CI [10.5, 20.0]) | `table1`, `table3b` |
| Reliable recourse — disadvantaged (full pool, n=785) | **0.127%** (1 of 785; exact upper bound 0.71%) | `table3_fullpool_reliability.csv` |
| Double Exclusion Index — advantaged vs. disadvantaged | **0.60 vs. 1.00** | `table5_dei.csv` |
| Action-horizon robustness (disadvantaged, τ up to ±50%) | ≤ 0.5% | `table8_horizon_sensitivity.csv` |
| Search-budget robustness (disadvantaged, 20 CFs/member) | 0.255% (2 of 785) | `table9_budget_robustness.csv` |
| Model-class robustness (CDR across 4 classifiers) | 6.3× – 17.3× | `table7_classifier_robustness.csv` |

Both axes remain directionally stable under every tested group-definition cutoff, action horizon, search budget, and model class, indicating the phenomenon is not an artifact of a single design choice.

---

## 3. Repository structure

```
.
├── data/                                 # raw source tables — NOT included (see §10)
│   └── .gitkeep
├── notebooks/
│   ├── 01_preprocessing_grouping_model.ipynb   # data → 12-feature governance → groups → model
│   ├── 02_recourse_generation.ipynb            # DiCE recourse (A/B/C), sample + full pool
│   ├── 03_metrics_reliability_dei.ipynb        # reliability (3 constraints), DEI, intervals
│   ├── 04_interaction_and_robustness.ipynb     # interaction vs accumulation; horizon/budget/model robustness
│   └── 05_make_figures.ipynb                   # grayscale figures (Figures 3–6)
├── results/
│   ├── figures/          # Figures 1–6 (PNG @600 dpi + PDF for 3–6)
│   ├── tables/           # all result tables as CSV (see §6)
│   └── cache/            # intermediate artifacts (git-ignored; written by 01–02)
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

Each notebook is a **single, self-contained code cell** with English comments and a notebook-name comment at the top. Paths are relative and resolved automatically whether a notebook is run from the repository root or from `notebooks/`; **no absolute paths are printed**, to preserve anonymity.

---

## 4. Method in brief

**Feature governance (12 features).** Every modeling variable is assigned, once and in advance, to one of three categories, which drive *both* the model and the recourse constraints:

- **Immutable `I` (5):** gender, age band, region, life stage, credit tenure. Advising a change is a violation.
- **Actionable `A` (6):** credit-usage amount (overall), credit-usage amount (industry-based), shopping spend, dining spend, leisure spend, merchant count. May change within a ±20% action horizon.
- **Non-controllable `N` (1):** issuer-assigned credit limit. Held fixed; any change is a violation.

**Recourse scenarios.** For every rejected member, counterfactual recourse is generated under three progressively stricter regimes:

| Scenario | Immutable | Actionable | Non-controllable |
|---|---|---|---|
| A — unconstrained | may vary | may vary | may vary |
| B — immutability | frozen | may vary | may vary |
| C — full guardrail | frozen | within ±20% | frozen |

**Reliability flags.** A generated path `x'` is **reliable** iff it triggers none of three flags: `v_imm` (any immutable feature changed), `v_act` (any actionable feature beyond ±20%, two-sided), `v_non` (the non-controllable feature changed).

**Reliability rates.**
- **PR-RR** (Pathway-Reliable): fraction of *all generated paths* that are reliable (conditional on generation).
- **BR-RR** (Borrower-Reliable): fraction of *members* with at least one reliable path.

**Double Exclusion Index.** For a group `S`,
`DEI(S) = P(rejected | S) × P(no reliable recourse | rejected, S)`,
an exact joint probability (chain rule; the second factor is conditional on rejection, so no independence assumption is made).

---

## 5. The pipeline, notebook by notebook

**`01_preprocessing_grouping_model.ipynb`**
*Inputs:* the three raw tables under `data/`.
*Does:* locates each table by its column-header signature; joins them on the member id; binarizes the VIP target; defines the intersectional groups with data-driven tenure cutoffs (25th/75th percentiles); applies the 12-feature governance encoding (nominal features integer-coded, age ordinal); trains a class-balanced Random Forest on a stratified sample with a held-out test split.
*Outputs:* `results/cache/nb1_artifacts.pkl` (model, splits, groups, partition, cutoffs, group-wise mean predicted probabilities); `results/tables/table0_model_metrics.csv`.

**`02_recourse_generation.ipynb`**
*Inputs:* `nb1_artifacts.pkl`.
*Does:* builds the DiCE generator with the nominal features declared categorical; generates up to four counterfactual recourse paths per rejected member under Scenarios A/B/C, for a sampled comparison (n=200 per group) and for the entire disadvantaged rejected pool.
*Outputs:* `results/cache/recourse_main.pkl`, `recourse_dis_full.pkl`, `nb2_frames.pkl`. This is the longest stage.

**`03_metrics_reliability_dei.ipynb`**
*Inputs:* the cache from 01–02.
*Does:* recomputes the three-constraint reliability directly from feature values; builds the recourse-rate table, the failure decomposition by constraint, and the full-population reliability with exact Clopper–Pearson bounds; computes bootstrap and Wilson intervals for the advantaged Scenario-C rate; computes the DEI with the empirical hard-rejection first factor.
*Outputs:* `table1`, `table2`, `table3`, `table3b`, `table5`.

**`04_interaction_and_robustness.ipynb`**
*Inputs:* the cache from 01–02.
*Does:* evaluates predicted approval across all eight attribute combinations and decomposes the response surface by effect-coded factorial regression on the log-odds scale (interaction vs. accumulation); sweeps the nine group-definition cutoffs for CDR; re-estimates the first-axis disparity under four model classes; and records the action-horizon and search-budget robustness checks (set `RUN_DICE = True` to regenerate the two DiCE-based checks).
*Outputs:* `table4`, `table4b`, `table6`, `table7`, `table8`, `table9`.

**`05_make_figures.ipynb`**
*Inputs:* the result tables.
*Does:* renders the four data figures (Figures 3–6) in grayscale with seaborn.
*Outputs:* `results/figures/fig1_scenario_tradeoff.{png,pdf}`, `fig2_reliable_rr_groups.{png,pdf}`, `fig3_cdr_sensitivity.{png,pdf}`, `fig4_dei_decomposition.{png,pdf}` (600 dpi).

---

## 6. Result tables

All tables are written to `results/tables/` as CSV.

| File | Contents |
|---|---|
| `table0_model_metrics.csv` | F1, recall, AUC-ROC on the held-out set |
| `table1_recourse_rates.csv` | RR, PR-RR, BR-RR by group × scenario (three-constraint reliability) |
| `table2_failure_breakdown.csv` | Per-constraint violation rates of generated paths (Scenarios A, B) |
| `table3_fullpool_reliability.csv` | Full disadvantaged pool (n=785): RR, reliable count, BR-RR, exact 95% upper bound |
| `table3b_advantaged_C_interval.csv` | Advantaged Scenario-C rate with bootstrap and Wilson 95% intervals |
| `table4_interaction_cells.csv` | Mean predicted approval across the eight attribute combinations |
| `table4b_interaction_decomposition.csv` | Effect-coded factorial decomposition (log-odds), incl. the three-way term |
| `table5_dei.csv` | Double Exclusion Index: P(reject), P(no reliable recourse), DEI by group |
| `table6_cdr_sensitivity.csv` | CDR across the nine group-definition cutoffs, with group sizes |
| `table7_classifier_robustness.csv` | First-axis disparity under four model classes |
| `table8_horizon_sensitivity.csv` | Disadvantaged Scenario-C reliable rate at τ ∈ {0.10, 0.20, 0.30, 0.50} |
| `table9_budget_robustness.csv` | Disadvantaged full-pool reliable rate at 4 vs. 20 counterfactuals/member |

---

## 7. Figures

Manuscript figure numbering:

- **Figure 1** — conceptual schematic of the double-exclusion mechanism (`fig0_double_exclusion_concept.png`).
- **Figure 2** — feature-governance partition and the three recourse scenarios (`fig_methodology_governance_scenarios.png`).
- **Figure 3** — nominal vs. reliable recourse across scenarios (`fig1_scenario_tradeoff`).
- **Figure 4** — reliable recourse by group with 95% intervals (`fig2_reliable_rr_groups`).
- **Figure 5** — Double Exclusion Index decomposition (`fig4_dei_decomposition`).
- **Figure 6** — CDR sensitivity heatmap (`fig3_cdr_sensitivity`).

Figures 3–6 are generated by notebook 05 (seaborn, grayscale, 600 dpi, PNG + PDF, captions omitted — captions live in the manuscript). Figures 1–2 are schematics prepared separately and placed in `results/figures/`. Note that the data-figure filenames (`fig1`–`fig4`) match the manuscript's `\includegraphics` keys and therefore differ by two from the printed figure numbers.

---

## 8. How to run

```bash
python -m venv venv && source venv/bin/activate      # or conda
pip install -r requirements.txt
jupyter notebook            # or jupyter lab
```

Run the notebooks in order: **01 → 02 → 03 → 04 → 05**. Notebook 01 is the expensive stage and caches its artifacts, so 03–05 can be re-run independently without repeating the model training. Notebook 02 (recourse generation over the full disadvantaged pool) is the longest stage. Notebooks can be run from the repository root or from inside `notebooks/` — paths resolve either way.

---

## 9. Configuration and reproducibility

- **Fixed seeds throughout** (`random_state = 42`; bootstrap seed fixed). Reported figures are stable across re-runs.
- **One governance partition** of twelve features (5 + 6 + 1) drives both the model and the recourse constraints, so an infeasible recommendation is defined once, before any recourse is generated, and is fully auditable.
- **Categorical-aware generation.** Nominal features (gender, region, life stage) are declared categorical to the counterfactual generator, so the search ranges over valid feature levels rather than fractional codes; the main and robustness analyses use one identical configuration.
- **Measured, not extrapolated.** The headline disadvantaged reliability bound is computed over the entire rejected pool, not inferred from a small sample.
- **`RUN_DICE` flag.** The two DiCE-based robustness checks in notebook 04 (action-horizon and search-budget sweeps) store their seed-fixed values by default; set `RUN_DICE = True` at the top of that notebook to regenerate them from the cached model (slower).
- **No identifying output.** Notebooks do not print absolute paths or machine-specific information.

---

## 10. Data access

The analysis uses a public synthetic financial (card-usage) dataset. The raw files are **not redistributed** in this repository. To reproduce:

1. Obtain the three required source tables from the original public provider:
   - **member information** (identified by a `VIP-grade` column),
   - **approval / sales** (identified by a recent credit-usage amount column),
   - **credit information** (identified by a card-usage-limit column).
2. Place the CSV files anywhere under `data/` (any subfolder structure is fine). Notebook 01 locates each table automatically by its column-header signature and uses the most recent monthly snapshot.
3. Run the notebooks in order.

> **Note.** Only the member / credit / approval tables are needed. External credit-bureau tables are intentionally **not** used, to keep the audit in a limited-information setting that isolates disparities produced by behavioral modeling itself.

---

## 11. Environment

Core dependencies (see `requirements.txt` for the full list):

`pandas`, `numpy`, `scikit-learn`, `dice-ml`, `scipy`, `seaborn`, `matplotlib`, `jupyter`.

The code targets Python 3.9+ and recent releases of the above. It is version-tolerant: a model cached under one scikit-learn version may emit an `InconsistentVersionWarning` when unpickled under another — this is a warning only, and training from scratch (the default end-to-end run) avoids it entirely.

---

## 12. Troubleshooting

- **A notebook can't find the data.** Ensure the three CSV files are somewhere under `data/`. Notebook 01 searches recursively and classifies files by header signature; a table with an unexpected header will be skipped.
- **`03`–`05` fail with a missing-file error.** Run `01` and `02` first; they populate `results/cache/`.
- **DiCE import error.** Install `dice-ml` (`pip install dice-ml`); it also pulls in its own dependencies.
- **Figure annotation error on very new pandas.** Notebook 05 avoids deprecated pandas calls; if you adapt it, prefer explicit array construction over `DataFrame.applymap`.
- **The DiCE robustness sweep is slow.** It is off by default (`RUN_DICE = False`) and uses stored seed-fixed values; enable it only to regenerate those two tables.

---

## 13. License and citation

Released under the MIT License (see `LICENSE`). Copyright is held by the anonymous authors during review.

A citation (BibTeX) and author/venue information will be added here upon acceptance. Until then, please cite the manuscript as an anonymized submission.
