# Targeted peptide panel — multimodal recurrence prediction

Combines a targeted DIA-MS peptide panel with the H&E AI recurrence-risk score (RS, from
`STAGE1_patch_classification.py` / `STAGE2_histogram_aggregation.ipynb`) into a multimodal
recurrence predictor, and evaluates it against the H&E-only model.

## What this notebook does

`targeted_peptide_panel.ipynb` runs the full targeted-panel analysis used in the paper:

1. **Load** the targeted DIA-MS precursor/peptide matrix and per-sample metadata.
2. **Select one representative precursor per gene** via inter-peptide correlation (IPC) — the
   precursor with the highest mean |Pearson r| against the other precursors of the same gene.
   This uses no outcome information, so it introduces no leakage.
3. **Build a single 13-gene tumor-associated composite score** from a fixed, pre-specified gene
   set (not re-selected or tuned on this cohort):
   - Up in high-risk regions (5): `EPPK1, MYH11, P4HA1, RPS15, YAP1`
   - Down in high-risk regions (8): `ARHGDIB, GET3, LAP3, LCP1, PSME2, SERPINB9, TYMP, WAS`

   This direction/gene set was defined from an independent spatial proteomics experiment (DEGs
   concordant across high- vs. low-risk regions in 2 patients), not from the outcome data used
   for evaluation below.
4. **Evaluate three models** by leave-one-out (LOO) and out-of-bag (OOB) bootstrap C-index /
   time-dependent AUC (2/3/5-year):
   - **M1** — H&E AI risk score (RS) only
   - **M2** — 13-gene composite score only
   - **M8** — RS + composite, combined by rank-sum
5. **Cox proportional-hazards** (univariate, direction-aligned, and multivariate) and **log-rank**
   tests (median split) for the composite, RS, and combined score.

There is **no tumor/immune split** and **no additional gene screening or intersection step** in
this analysis — the 13 genes above are used as-is.

## Data you need to supply

This repo ships no proteomics data. Place the following under `data/targeted_proteomics/`
(relative to this notebook):

- `target_final_pep_with_scores3.tsv` — targeted-panel peptide/precursor table with per-sample
  intensity columns, gene annotation, and a recurrence-risk score column used to build `RS`.
- `report.pr_matrix.tsv` — DIA-MS precursor-level report matrix (e.g. Spectronaut/DIA-NN style
  `pr_matrix` export) covering the same samples.
- `column_check.txt` — sample-ID/column-order mapping between the two files above.

None of these files are included here — they contain patient-derived proteomic measurements.

## Output

Running the notebook reproduces, for the tumor-composite panel:
- LOO and OOB C-index / time-dependent AUC for M1 (H&E AI risk only), M2 (13-gene composite
  only), and M8 (RS + composite, rank-sum)
- `OOB_absolute_model_performance_table.csv` — the OOB absolute-performance summary table
- Univariate/multivariate Cox PH hazard ratios and log-rank test results

## Limitations

Evaluation is LOO/OOB on a single, small cohort (n≈49 with proteomic data available) — see the
paper for the intended interpretation of these metrics. As with the STAGE1/STAGE2 H&E pipeline,
this analysis is aimed at biomarker discovery and characterizing the AI-guided high-risk regions,
not at establishing a validated, deployable multimodal predictor.
