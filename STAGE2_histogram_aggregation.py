"""
STAGE2 -- Top-10 + 20-bin histogram + Lasso

Aggregates per-patch recurrence-risk scores (RRS, from `compute_patch_rrs.py`) into a
patient-level prediction, using the configuration reported in the paper:

- Select each patient's top-10 highest-RRS patches
- Represent those 10 values as a 20-bin histogram (range 0-1, normalized by count/10)
- Fit L1-penalized logistic regression ("Lasso") on the train split, evaluate on the test split
- Report bootstrap AUC and C-index (using RFS + Recur) on the test set

See the repo README's Limitations section: this stage's hyperparameters (top-k, bin count) were
selected against this same test split rather than via a train-only nested search -- treat the
reported test metrics as descriptive of the configuration the paper used, not as an unbiased
estimate of a validated predictor's generalization performance.

Outputs (predictions + metrics) are written to --out_dir as CSV/text rather than notebook cell
output, since per-patient predictions carry a patient identifier column ('tube label') that
shouldn't be baked into a committed, rendered file.
"""
import argparse
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='STAGE2: histogram aggregation + Lasso on per-patch RRS.')
parser.add_argument('--rrs_pkl', type=str, default=os.path.join(REPO_DIR, 'slide_prob_scores_per_patient_iter9000.pkl'))
parser.add_argument('--train_df_dir', type=str, default=os.path.join(REPO_DIR, '0_folds', 'TNBC_train_df.csv'))
parser.add_argument('--test_df_dir', type=str, default=os.path.join(REPO_DIR, '0_folds', 'TNBC_test_df.csv'))
parser.add_argument('--out_dir', type=str, default=os.path.join(REPO_DIR, 'stage2_outputs'))
parser.add_argument('--top_k', type=int, default=10)
parser.add_argument('--num_bins', type=int, default=20)
parser.add_argument('--n_bootstrap', type=int, default=2000)
parser.add_argument('--ci', type=float, default=95)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()


def bootstrap_auc_ci(y_true, y_score, n_bootstrap, ci, random_state):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.RandomState(random_state)
    n = len(y_true)

    auc = roc_auc_score(y_true, y_score)

    boots = []
    for _ in range(n_bootstrap):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        boots.append(roc_auc_score(yt, ys))
    boots = np.array(boots)

    alpha = (100 - ci) / 2
    lower, upper = np.percentile(boots, [alpha, 100 - alpha])
    return auc, lower, upper


def bootstrap_auc_cindex_ci(y_true, y_score, survival_time, event_observed, n_bootstrap, ci, random_state):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    survival_time = np.asarray(survival_time, dtype=float)
    event_observed = np.asarray(event_observed, dtype=int)

    rng = np.random.RandomState(random_state)
    n = len(y_true)

    auc = roc_auc_score(y_true, y_score)
    # a higher risk score means higher recurrence risk, so negate it for concordance_index
    cindex = concordance_index(survival_time, -y_score, event_observed)

    boot_aucs, boot_cidx = [], []
    for _ in range(n_bootstrap):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        yt, ys, st, ev = y_true[idx], y_score[idx], survival_time[idx], event_observed[idx]
        if len(np.unique(yt)) >= 2:
            boot_aucs.append(roc_auc_score(yt, ys))
        try:
            c = concordance_index(st, -ys, ev)
            if np.isfinite(c):
                boot_cidx.append(c)
        except (ZeroDivisionError, ValueError):
            pass

    boot_aucs, boot_cidx = np.array(boot_aucs), np.array(boot_cidx)
    alpha = (100 - ci) / 2
    auc_lo, auc_hi = np.percentile(boot_aucs, [alpha, 100 - alpha])
    c_lo, c_hi = np.percentile(boot_cidx, [alpha, 100 - alpha])

    return {
        'auc': auc, 'auc_lower': auc_lo, 'auc_upper': auc_hi,
        'cindex': cindex, 'cindex_lower': c_lo, 'cindex_upper': c_hi,
        'n_valid_auc': len(boot_aucs), 'n_valid_cindex': len(boot_cidx),
    }


def build_features(df_original, prob_dict, top_k, num_bins):
    labels, feats, names = [], [], []
    for key in sorted(df_original['tube label']):
        scores = prob_dict.get(key)
        if not scores:
            print(f'no RRS for {key}, skipping')
            continue
        top_scores = sorted(scores, reverse=True)[:top_k]
        hist, _ = np.histogram(top_scores, bins=num_bins, range=(0, 1))
        hist_norm = hist / hist.sum()

        label = df_original[df_original['tube label'] == key]['Recur'].item()
        labels.append(label)
        feats.append(hist_norm)
        names.append(key)
    return names, np.array(feats), np.array(labels)


def main():
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.rrs_pkl, 'rb') as f:
        slide_prob_scores_per_patient = pickle.load(f)

    train_df_original = pd.read_csv(args.train_df_dir, encoding='cp949')
    test_df_original = pd.read_csv(args.test_df_dir, encoding='cp949')

    print(f'RRS available for {len(slide_prob_scores_per_patient)} patients')
    print(f'train: {len(train_df_original)}, test: {len(test_df_original)}')

    train_names, X_train, y_train = build_features(train_df_original, slide_prob_scores_per_patient, args.top_k, args.num_bins)
    test_names, X_test, y_test = build_features(test_df_original, slide_prob_scores_per_patient, args.top_k, args.num_bins)

    print('train:', X_train.shape, Counter(y_train))
    print('test:', X_test.shape, Counter(y_test))

    logit_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('logit', LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', random_state=args.seed)),
    ])
    logit_pipe.fit(X_train, y_train)

    y_proba_train = logit_pipe.predict_proba(X_train)[:, 1]
    y_proba_test = logit_pipe.predict_proba(X_test)[:, 1]

    train_pred_df = pd.DataFrame({'tube label': train_names, 'risk_score': y_proba_train})
    test_pred_df = pd.DataFrame({'tube label': test_names, 'risk_score': y_proba_test})

    train_pred_df = train_pred_df.merge(train_df_original[['tube label', 'Recur', 'RFS']], on='tube label', how='left')
    test_pred_df = test_pred_df.merge(test_df_original[['tube label', 'Recur', 'RFS']], on='tube label', how='left')

    train_pred_path = os.path.join(args.out_dir, 'stage2_top10_lasso_train_predictions.csv')
    test_pred_path = os.path.join(args.out_dir, 'stage2_top10_lasso_test_predictions.csv')
    train_pred_df.to_csv(train_pred_path, index=False)
    test_pred_df.to_csv(test_pred_path, index=False)
    print(f'\nWrote per-patient predictions to {train_pred_path} and {test_pred_path}')

    results = bootstrap_auc_cindex_ci(
        y_true=test_pred_df['Recur'], y_score=test_pred_df['risk_score'],
        survival_time=test_pred_df['RFS'], event_observed=test_pred_df['Recur'],
        n_bootstrap=args.n_bootstrap, ci=args.ci, random_state=args.seed,
    )
    train_auc, train_lo, train_hi = bootstrap_auc_ci(y_train, y_proba_train, args.n_bootstrap, args.ci, args.seed)

    metrics_lines = [
        f"n_train={len(train_pred_df)}, n_test={len(test_pred_df)}, test_recurrence={int(test_pred_df['Recur'].sum())}",
        f"Train AUC (fit set, for reference): {train_auc:.3f} (95% CI: {train_lo:.3f}-{train_hi:.3f})",
        f"Test AUC:     {results['auc']:.3f} (95% CI: {results['auc_lower']:.3f}-{results['auc_upper']:.3f})",
        f"Test C-index: {results['cindex']:.3f} (95% CI: {results['cindex_lower']:.3f}-{results['cindex_upper']:.3f})",
    ]
    metrics_path = os.path.join(args.out_dir, 'stage2_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write('\n'.join(metrics_lines) + '\n')

    print()
    print('\n'.join(metrics_lines))
    print(f'\nWrote summary metrics to {metrics_path}')


if __name__ == '__main__':
    main()
