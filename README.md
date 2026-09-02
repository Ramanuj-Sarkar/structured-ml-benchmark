# structured-ml-benchmark — Give Me Some Credit

Proper feature engineering + cross-validation benchmark of **XGBoost vs
LightGBM vs CatBoost** on the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)
dataset (Kaggle, 150k rows, 6.7% default rate), with **SHAP explainability**
and a **model card**.

## Results (headline)

| Model                          | Test ROC-AUC | 5-fold CV ROC-AUC (mean ± std) |
|--------------------------------|--------------|--------------------------------|
| **CatBoost** (winner)          | **0.8702**   | 0.8651 ± 0.0034                |
| XGBoost                        | 0.8699       | 0.8648 ± 0.0035                |
| LightGBM                       | 0.8694       | 0.8636 ± 0.0030                |
| Logistic Regression (baseline) | 0.8639       | —                              |

All three GBDTs clearly beat the linear baseline and land within ~0.001 of
each other on the held-out test set (30k rows); CatBoost is the best on both
CV and test, and by a hair on PR-AUC (0.412 vs 0.411/0.404). Full numbers
(AUC / PR-AUC / KS / Gini / Brier / log loss), tuned hyperparameters, SHAP
rankings, and the decision threshold are in
[`results/model_card.md`](results/model_card.md) and
[`results/metrics.json`](results/metrics.json).

## Methodology

1. **Fold-safe preprocessing** — all cleaning statistics (winsorization caps,
   imputation medians) are fit *inside* every CV fold / test refit by
   `PreprocessTransformer` (`src/data_prep.py`), so there is no leakage:
   - 96/98 sentinel errors in the three past-due columns → missing + flag
   - `age` 0 / >100 → clip + flag
   - `MonthlyIncome` NaN/0 (~21%) → log + age-decade median imputation + flag
   - `NumberOfDependents` NaN (~2.6%) → median + flag
   - `RevolvingUtilization` / `DebtRatio` extreme outliers → winsorize
   - 27 engineered features incl. delinquency aggregates, utilization
     interactions, income-per-dependent, income-to-age, buckets & flags
2. **Honest tuning** — 80/20 stratified holdout; the test set is used exactly
   once. Each GBDT is tuned with random search over out-of-fold AUC, where
   every candidate is trained with **early stopping** on an inner validation
   split (sklearn's `RandomizedSearchCV` cannot do this per fold).
3. **Evaluation** — 5-fold stratified CV with tuned hyperparameters and
   early stopping **per fold** (`results/cv_fold_metrics.csv`,
   `figures/cv_auc_boxplot.png`), plus a final refit on 100% of the training
   split (tree count chosen by an early-stopping run on a 90/10 split — no
   training rows wasted) evaluated on the untouched test set (ROC / PR /
   calibration curves). A balanced logistic-regression baseline (in a
   scikit-learn `Pipeline`) provides context.
4. **Explainability** — SHAP `TreeExplainer` on a seeded 2 000-row test
   subsample: beeswarm + mean-|SHAP| per model, dependence plots for the top
   features of the best model, and a cross-model importance chart.

## Quickstart

```bash
python -m venv .venv-ml && source .venv-ml/bin/activate
pip install -r requirements.txt

# full pipeline: preprocessing -> tuning -> CV -> test eval -> SHAP
python -m src.pipeline

# or step by step
python src/train.py       # tuning, CV, test evaluation, figures, metrics.json
python src/explain.py     # SHAP figures + shap_summary.json
python src/model_card.py  # render results/model_card.md
```

Runtime ≈ 15–20 min on a 12-core laptop (early stopping keeps it fast).

## Layout

```
data/structured-ml-dataset.csv   raw data (unchanged)
src/data_prep.py                 fold-safe cleaning + feature engineering
src/train.py                     tuning, 5-fold CV, test eval, plots
src/explain.py                   SHAP analysis
src/model_card.py                renders the model card
src/pipeline.py                  end-to-end entry point
results/metrics.json             all numbers (machine-readable)
results/cv_fold_metrics.csv      per-fold metrics
results/model_card.md            the model card
results/figures/*.png            ROC/PR/calibration/CV/SHAP figures
results/models/*.joblib          fitted prep + models
```
