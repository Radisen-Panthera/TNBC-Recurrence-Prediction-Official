# TNBC Recurrence Prediction from H&E WSIs

Weakly-supervised patch classification + histogram aggregation pipeline for predicting
triple-negative breast cancer (TNBC) recurrence from H&E whole-slide images (WSIs), and for
generating spatially-resolved recurrence-risk heatmaps. This is a cleaned-up, runnable version of
the pipeline used in *"Spatial proteomics guided by H&E-based AI reveals recurrence-risk niches in
triple-negative breast cancer"*.

The AI model here is **not** intended as a validated, deployable recurrence predictor. It is an
exploratory tool: a patch-level classifier trained with weak (patient-level) labels is used to
generate a risk-score heatmap, which is aggregated into a patient-level score and also used to
physically guide spatial proteomic sampling of high- vs. low-risk tumor regions. STAGE1/STAGE2 use
a single train/test split (no cross-validation) — see the paper for the intended interpretation of
the reported AUC/C-index.

## Pipeline overview

```
TIGER_training/prepare_tiger_tiles.py            (0) extract 512x512 labeled tiles from a raw
            │                                          TIGER challenge download
            ▼
TIGER_training/training_TIGER_classifier.ipynb   (1) train a tissue-compartment (ROI) classifier
            │                                          on the TIGER dataset (train/val/test split;
            │                                          checkpoint selected on val, reported once on
            │                                          held-out test)
            ▼
TIGER_training/extract_candidate_patches.py     (1.5) tile your own WSI cohort into candidate
            │                                          tissue-patch coordinates (foreground masking
            │                                          + non-overlapping grid)
            ▼
TIGER_training/labeling_TIGER_inference.py       (2) run the ROI classifier on your WSI cohort,
            │                                          filter to tumor / TAS / necrosis / inflamed
            │                                          stroma patches
            ▼
patch_info_extraction.ipynb                      (3) build the per-slide patch coordinate/label
            │                                          dictionary used for STAGE1 training
            ▼
STAGE1_patch_classification.py                   (4) weakly-supervised patch-level recurrence-risk
            │                                          classifier (single train/test split, no CV —
            │                                          this stage is for biomarker discovery /
            │                                          heatmap generation, not a final predictor)
            ▼
compute_patch_rrs.py                             (5) apply a trained STAGE1 checkpoint to every ROI
            │                                          patch, producing per-patch recurrence-risk
            │                                          scores (RRS) for every patient
            ▼
STAGE2_histogram_aggregation.py                  (6) aggregate each patient's top-k highest-RRS
                                                       patches into a histogram feature, fit a
                                                       simple model (Lasso/ElasticNet) to predict
                                                       recurrence, report test AUC / C-index
            ▼
proteomics_analysis/targeted_panel/              (7) combine a targeted DIA-MS peptide panel with
                                                       the RRS-based risk score (RS) into a
                                                       multimodal recurrence predictor
```

Stages (1)-(6) use only H&E imaging. Stage (7), in `proteomics_analysis/`, is a separate,
optional extension that adds proteomic data — see its own README for details.

## Repository layout

```
dataset.py                    PathologyPatchDataset, DINO-style augmentation, stain normalization
model.py                      ConvNeXt-based backbone/classifier, LoRA/SoMA fine-tuning
TIGER_training/
  prepare_tiger_tiles.py            Stage (0)
  training_TIGER_classifier.ipynb   Stage (1)
  extract_candidate_patches.py      Stage (1.5)
  labeling_TIGER_inference.py       Stage (2)
patch_info_extraction.ipynb         Stage (3)
STAGE1_patch_classification.py      Stage (4)
compute_patch_rrs.py                Stage (5)
STAGE2_histogram_aggregation.py     Stage (6)
proteomics_analysis/
  targeted_panel/                   Stage (7) — see its own README
```

## Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU for training/inference (no CPU fallback). Tested with Python 3.11 / PyTorch
with CUDA 12.x.

## Data you need to supply

This repo ships **no data** — no WSIs, no clinical labels, no model checkpoints. You need:

- **WSIs** in a directory openslide can read (`.svs` etc.), one file per patient.
- **A DINO-pretrained ConvNeXt-Base checkpoint** (`G2B_BRCA.pth` in the scripts' defaults) used as
  the frozen/fine-tuned feature-extractor backbone, with a `state_dict['student']` key holding a
  `StudentModel_convnext`-shaped state dict (backbone + projection head).
- **Clinical/outcome CSVs** with at minimum `tube label` (matches the WSI filename stem),
  `Recur` (0/1), and `RFS` (recurrence-free survival time, for C-index) columns. STAGE1/STAGE2
  expect a pre-split `train_df` / `test_df` pair (see `--train_df_dir`/`--test_df_dir` in
  `STAGE1_patch_classification.py` and `compute_patch_rrs.py`).
- **A raw TIGER challenge download** ([tiger.grand-challenge.org](https://tiger.grand-challenge.org/),
  released under CC BY-NC 4.0) — specifically the `wsirois/roi-level-annotations/tissue-bcss/`
  ROI images + tissue-compartment masks, used by `TIGER_training/prepare_tiger_tiles.py` (stage 0)
  to build the 512x512 tile dataset for training the ROI classifier in stage (1).

All paths above are **machine-specific argparse defaults** in the scripts — override them via CLI
flags (or edit the defaults) for your environment; don't assume the shipped defaults are valid
paths on your machine.

## Running it

```bash
# (0) extract 512x512 labeled tiles from a raw TIGER download (produces train/val/test CSVs)
python TIGER_training/prepare_tiger_tiles.py \
  --images_dir /path/to/your/TIGER/wsirois/roi-level-annotations/tissue-bcss/images \
  --masks_dir /path/to/your/TIGER/wsirois/roi-level-annotations/tissue-bcss/masks \
  --output_dir ./TIGER_training/TIGER_labeled_tiles_512

# (1) train the ROI classifier — run cells in TIGER_training/training_TIGER_classifier.ipynb

# (1.5) tile your own WSI cohort into candidate tissue-patch coordinates
python TIGER_training/extract_candidate_patches.py \
  --slide_dir /path/to/wsi \
  --output_dir ./TIGER_training/candidate_patch_coords

# (2) label your WSI cohort with the trained ROI classifier
python TIGER_training/labeling_TIGER_inference.py \
  --coords_dir ./TIGER_training/candidate_patch_coords \
  --slide_dir /path/to/wsi \
  --clinical_dir /path/to/clinical.csv \
  --output_dir ./ROI_sampling_all

# (3) build the patch_info pickle used by STAGE1 — run patch_info_extraction.ipynb

# (4) train STAGE1
python STAGE1_patch_classification.py \
  --dirs ./ROI_sampling_all/coords \
  --slide_dir /path/to/wsi \
  --train_df_dir ./0_folds/train_df.csv \
  --test_df_dir ./0_folds/test_df.csv \
  --train_img_pkl ./patch_info_KBSMC_train_zero_filtered.pkl \
  --gpu 0

# (5) compute per-patch risk scores from a trained checkpoint
python compute_patch_rrs.py \
  --checkpoint ./weights_STAGE1/best_model.pth \
  --coords_dir ./ROI_sampling_all/coords \
  --slide_dir /path/to/wsi \
  --gpu 0

# (6) aggregate to patient-level risk and evaluate
python STAGE2_histogram_aggregation.py \
  --rrs_pkl ./slide_prob_scores_per_patient_iter9000.pkl \
  --train_df_dir ./0_folds/TNBC_train_df.csv \
  --test_df_dir ./0_folds/TNBC_test_df.csv
```

`STAGE2_histogram_aggregation.py` writes per-patient predictions and summary metrics (AUC,
C-index) to `--out_dir` (default `./stage2_outputs/`) rather than printing per-patient identifiers
into a committed notebook — see Limitations below.

`STAGE1_patch_classification.py` validates every `--test_interval` iterations on the held-out
test set (Val/AUC, KS-statistic, Wasserstein distance, logged to TensorBoard under `--writer_dir`)
and supports early stopping (`--early_stop_patience`) and resuming from a checkpoint
(`--resume_checkpoint` / `--start_iteration` / `--best_auc_init`) so you don't have to restart
training from scratch to extend a run.

## Limitations

STAGE1 (patch-level weakly-supervised training) and STAGE2 (histogram aggregation) use a single
train/test split rather than cross-validation — this pipeline is designed for biomarker discovery
and risk-heatmap generation, not as a validated, deployable recurrence predictor. See the paper for
the intended scope and interpretation of the reported metrics.

The TIGER ROI classifier (stage 1) is the one place in this pipeline with a proper train/validation/
test split: the validation split (not the test split) is used for checkpoint selection during
training, and the test split is evaluated exactly once, after training, using the checkpoint the
validation loss selected. This is different from STAGE1/STAGE2 above, where the same held-out split
is used both to pick the reported checkpoint/configuration and to report its performance — treat
the TIGER classifier's test metrics as an unbiased estimate of its tissue-compartment classification
accuracy, and STAGE1/STAGE2's test metrics as descriptive of the specific configuration reported in
the paper rather than as validated generalization estimates.
