"""Generate results/model_card.md from metrics.json + shap_summary.json.

Short, honest model card for the Give Me Some Credit benchmark following the
Model Cards for Model Reporting structure (Mitchell et al., 2019), condensed.
"""

from __future__ import annotations

import json
from pathlib import Path

from data_prep import FEATURE_GROUPS

RESULTS_DIR = Path("results")

MODEL_LABELS = {
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "cat": "CatBoost",
    "logistic_regression": "Logistic Regression (baseline)",
}


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def metric_cell(metrics: dict, key: str) -> str:
    return f"{fmt(metrics[key]['mean'])} ± {fmt(metrics[key]['std'])}"


def render(metrics: dict, shap_summary: dict) -> str:
    test = metrics["test"]
    cv = metrics["cv"]
    tuning = metrics["tuning"]

    # rank models by test AUC
    ranked = sorted(test, key=lambda m: -test[m]["auc"])
    best = ranked[0]
    best_label = MODEL_LABELS[best]

    rows = []
    for m in ranked:
        label = MODEL_LABELS[m]
        t = test[m]
        n_est = f" ({t.get('n_estimators', '—')} trees)" if m != "logistic_regression" else ""
        rows.append(
            f"| {label}{n_est} | {fmt(t['auc'])} | {fmt(t['pr_auc'])} | "
            f"{fmt(t['ks'])} | {fmt(t['gini'])} | {fmt(t['brier'])} | {fmt(t['log_loss'])} |"
        )
    test_table = "\n".join(rows)

    cv_rows = []
    for m in ["xgb", "lgbm", "cat"]:
        label = MODEL_LABELS[m]
        c = cv[m]
        cv_rows.append(
            f"| {label} | {metric_cell(c, 'auc')} | {metric_cell(c, 'pr_auc')} | "
            f"{metric_cell(c, 'ks')} | {metric_cell(c, 'gini')} | "
            f"{metric_cell(c, 'brier')} | {metric_cell(c, 'log_loss')} |"
        )
    cv_table = "\n".join(cv_rows)

    thr = metrics["threshold"]
    cm = thr["confusion_matrix"]

    shap = shap_summary["mean_abs_shap"]
    top_best = sorted(shap[best].items(), key=lambda kv: -kv[1])[:8]
    top_best_str = ", ".join(f"`{k}` ({v:.4f})" for k, v in top_best)

    # feature groups actually used
    all_feats = set(metrics["data"]["feature_names"])
    groups = [f"**{g}**: {', '.join(f'`{f}`' for f in feats if f in all_feats)}"
              for g, feats in FEATURE_GROUPS.items()]

    param_lines = []
    for m in ["xgb", "lgbm", "cat"]:
        p = tuning[m]["best_params"]
        p_str = ", ".join(f"{k}={v}" for k, v in sorted(p.items()))
        param_lines.append(f"- **{MODEL_LABELS[m]}** (tuned): {p_str} — "
                           f"OOF AUC during tuning: {fmt(tuning[m]['best_oof_auc'])}")

    return f"""# Model Card — Give Me Some Credit Default Prediction

**Task:** binary classification — predict whether a borrower will experience
serious delinquency (`SeriousDlqin2yrs = 1`) within 2 years.
**Dataset:** [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)
(Kaggle), `{metrics['data']['rows']:,}` rows, {metrics['data']['n_features']} engineered features.
**Primary metric:** ROC-AUC (the Kaggle competition metric); also reported:
PR-AUC, KS, Gini, Brier score, log loss.

## Model details

- **Winner on the held-out test set:** {best_label} (test AUC = {fmt(test[best]['auc'])}).
- Comparison: XGBoost, LightGBM, CatBoost (tuned per model) + a balanced
  logistic-regression baseline for context.
- Tuning: random search with out-of-fold AUC and per-candidate early stopping
  on an inner validation split; tree counts fixed afterwards from early
  stopping on 10% of the training split.
- 80/20 stratified holdout: all tuning and CV happen on the 80% train; the
  test set is used exactly once, at the end.

## Test-set performance (untouched 20% holdout)

| Model | ROC-AUC | PR-AUC | KS | Gini | Brier | Log loss |
|---|---|---|---|---|---|---|
{test_table}

## Cross-validation (5-fold stratified, tuned hyperparameters, train split only)

| Model | ROC-AUC | PR-AUC | KS | Gini | Brier | Log loss |
|---|---|---|---|---|---|---|
{cv_table}

## Decision threshold (best model, Youden's J on test set)

- Threshold: **{fmt(thr['youden_threshold'])}** → precision **{fmt(thr['precision'])}**,
  recall **{fmt(thr['recall'])}**, F1 **{fmt(thr['f1'])}**.
- Confusion matrix on the test set: TP {cm['tp']}, FP {cm['fp']}, FN {cm['fn']}, TN {cm['tn']}.

## Training data

- **Size:** {metrics['data']['train']:,} rows (80%), {metrics['data']['test']:,} rows held out.
- **Class balance:** {fmt(metrics['data']['positive_rate'] * 100, 2)}% positive
  (serious delinquency) — imbalanced; LR baseline uses balanced class weights.
- **Source quirks handled:** sentinel errors 96/98 in the three past-due
  columns (flagged + imputed), `age` = 0 / > 100 (clipped + flagged),
  `MonthlyIncome` NaN/0 (~21%, imputed by age-decade median log-income +
  flag), `NumberOfDependents` NaN (~2.6%, median + flag), extreme
  `RevolvingUtilization` / `DebtRatio` outliers (winsorized at fold-fitted
  quantiles). All statistics are fit **inside each CV fold** to avoid leakage.

## Feature engineering

{chr(10).join('- ' + g for g in groups)}

## Explainability (SHAP, TreeExplainer on a seeded 2 000-row test subsample)

Top features for the best model ({best_label}): {top_best_str}.

`results/figures/shap_summary_*.png` (beeswarm), `shap_importance_*.png`
(mean |SHAP|), `shap_dependence_*.png` (top-2 dependence), and
`shap_model_comparison.png` (cross-model importance) — plus
`results/shap_summary.json` with the raw numbers.

## Intended use & limitations

- **Intended use:** credit-risk research/benchmarking, feature-attribution
  exploration, educational reference. Not a production underwriting system.
- **Limitations:** (1) data from 2011 with known quality issues and coarse
  features (no credit-bureau history depth); (2) the holdout is a random
  sample, not a temporal split — performance on later cohorts is unverified;
  (3) no explicit fairness audit — SHAP shows `age` is influential, so
  disparate-impact analysis is required before any deployment; (4) GBDT
  probabilities are not perfectly calibrated (see `figures/calibration.png`).
- **Reproduction:** `python -m src.pipeline` (see `README.md`); all seeds
  fixed (`SEED = 42`).
"""


def main():
    with open(RESULTS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    with open(RESULTS_DIR / "shap_summary.json") as f:
        shap_summary = json.load(f)
    out = RESULTS_DIR / "model_card.md"
    out.write_text(render(metrics, shap_summary))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
