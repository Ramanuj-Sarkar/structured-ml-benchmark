# Model Card — Give Me Some Credit Default Prediction

**Task:** binary classification — predict whether a borrower will experience
serious delinquency (`SeriousDlqin2yrs = 1`) within 2 years.
**Dataset:** [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)
(Kaggle), `150,000` rows, 27 engineered features.
**Primary metric:** ROC-AUC (the Kaggle competition metric); also reported:
PR-AUC, KS, Gini, Brier score, log loss.

## Model details

- **Winner on the held-out test set:** CatBoost (test AUC = 0.8702).
- Comparison: XGBoost, LightGBM, CatBoost (tuned per model) + a balanced
  logistic-regression baseline for context.
- Tuning: random search with out-of-fold AUC and per-candidate early stopping
  on an inner validation split. Final CV uses early stopping **per fold**;
  the test-set models are refit on 100% of the training split with the tree
  count chosen by an early-stopping run on a 90/10 split of that training
  data (no training rows wasted).
- 80/20 stratified holdout: all tuning and CV happen on the 80% train; the
  test set is used exactly once, at the end.

## Test-set performance (untouched 20% holdout)

| Model                          | ROC-AUC | PR-AUC | KS     | Gini   | Brier  | Log loss |
|--------------------------------|---------|--------|--------|--------|--------|----------|
| CatBoost (1066 trees)          | 0.8702  | 0.4119 | 0.5808 | 0.7404 | 0.0485 | 0.1754   |
| XGBoost (484 trees)            | 0.8699  | 0.4106 | 0.5846 | 0.7399 | 0.0485 | 0.1755   |
| LightGBM (53 trees)            | 0.8694  | 0.4043 | 0.5838 | 0.7389 | 0.0487 | 0.1761   |
| Logistic Regression (baseline) | 0.8639  | 0.3947 | 0.5741 | 0.7278 | 0.1447 | 0.4682   |

## Cross-validation (5-fold stratified, tuned hyperparameters, train split only)

| Model    | ROC-AUC         | PR-AUC          | KS              | Gini            | Brier           | Log loss        |
|----------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| XGBoost  | 0.8648 ± 0.0035 | 0.4018 ± 0.0052 | 0.5770 ± 0.0069 | 0.7296 ± 0.0070 | 0.0490 ± 0.0003 | 0.1777 ± 0.0013 |
| LightGBM | 0.8636 ± 0.0030 | 0.4003 ± 0.0063 | 0.5765 ± 0.0067 | 0.7272 ± 0.0059 | 0.0491 ± 0.0003 | 0.1782 ± 0.0012 |
| CatBoost | 0.8651 ± 0.0034 | 0.4038 ± 0.0050 | 0.5786 ± 0.0087 | 0.7301 ± 0.0069 | 0.0490 ± 0.0002 | 0.1775 ± 0.0012 |

## Decision threshold (best model, Youden's J on test set)

- Threshold: **0.0692** → precision **0.2203**,
  recall **0.7781**, F1 **0.3433**.
- Confusion matrix on the test set: TP 1560, FP 5522, FN 445, TN 22473.

## Training data

- **Size:** 120,000 rows (80%), 30,000 rows held out.
- **Class balance:** 6.68% positive
  (serious delinquency) — imbalanced; LR baseline uses balanced class weights.
- **Source quirks handled:** sentinel errors 96/98 in the three past-due
  columns (flagged + imputed), `age` = 0 / > 100 (clipped + flagged),
  `MonthlyIncome` NaN/0 (~21%, imputed by age-decade median log-income +
  flag), `NumberOfDependents` NaN (~2.6%, median + flag), extreme
  `RevolvingUtilization` / `DebtRatio` outliers (winsorized at fold-fitted
  quantiles). All statistics are fit **inside each CV fold** to avoid leakage.

## Feature engineering

- **credit utilization**: `RevolvingUtilizationOfUnsecuredLines`, `utilization_bucket`, `high_utilization_flag`, `utilization_x_income`, `utilization_x_debt`
- **debt burden**: `DebtRatio`, `debt_income_ratio_flag`, `income_to_age`
- **income**: `MonthlyIncome_log`, `income_missing_flag`, `income_per_dependent`
- **delinquency history**: `NumberOfTime30-59DaysPastDueNotWorse`, `NumberOfTimes90DaysLate`, `NumberOfTime60-89DaysPastDueNotWorse`, `total_delinquency`, `max_delinquency`, `delinquency_rate`, `any_delinquency_flag`, `pd_sentinel_flag`
- **accounts / credit lines**: `NumberOfOpenCreditLinesAndLoans`, `NumberRealEstateLoansOrLines`, `real_estate_share`
- **demographics**: `age`, `age_decade`, `age_outlier_flag`, `NumberOfDependents`, `dependents_missing_flag`

## Explainability (SHAP, TreeExplainer on a seeded 2 000-row test subsample)

Top features for the best model (CatBoost): `utilization_x_income` (0.2133), `RevolvingUtilizationOfUnsecuredLines` (0.1911), `utilization_bucket` (0.1646), `NumberOfOpenCreditLinesAndLoans` (0.1608), `age` (0.1500), `delinquency_rate` (0.1485), `any_delinquency_flag` (0.1221), `NumberOfTimes90DaysLate` (0.0992).

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
