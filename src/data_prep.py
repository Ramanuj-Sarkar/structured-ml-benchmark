"""Data cleaning + feature engineering for the Give Me Some Credit dataset.

Design principles
-----------------
* Every transform is *fold-safe*: all fitted statistics (winsorization caps,
  imputation medians) are learned from the training portion only, inside the
  CV loop, via a scikit-learn compatible transformer (`PreprocessTransformer`).
* The transformer emits a dense, NaN-free numeric matrix so the exact same
  features feed every model (XGBoost / LightGBM / CatBoost / Logistic
  Regression baseline), keeping the comparison apples-to-apples.
* Missingness is *not* silently discarded: explicit flag columns are added so
  tree models can still exploit "income unknown / dependents unknown" signals.

Known data-quality issues in Give Me Some Credit (this file)
------------------------------------------------------------
* ``NumberOfTime30-59DaysPastDueNotWorse``, ``NumberOfTimes90DaysLate`` and
  ``NumberOfTime60-89DaysPastDueNotWorse`` contain sentinel errors 96 and 98
  (264 + 5 rows); treated as missing + flagged.
* ``age`` contains 0 (1 row) and >100 (13 rows); clipped + flagged.
* ``MonthlyIncome`` is NaN in ~19.8% and 0 in ~1.1% of rows; imputed with the
  median log-income of the same age decade, plus a flag.
* ``NumberOfDependents`` is NaN in ~2.6% of rows; median-imputed + flagged.
* ``RevolvingUtilizationOfUnsecuredLines`` and ``DebtRatio`` have extreme
  outliers (max 50 708 / 329 664); winsorized at fold-fitted quantiles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DATA_PATH = "data/structured-ml-dataset.csv"

TARGET = "SeriousDlqin2yrs"
ID_COL = "Unnamed: 0"

PAST_DUE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]
PAST_DUE_SENTINELS = (96, 98)

# Column -> quantile used for winsorization (fit on training folds only).
# MonthlyIncome and NumberOfDependents are handled in log/median space instead.
WINSORIZE_SPEC = {
    "RevolvingUtilizationOfUnsecuredLines": 0.995,
    "DebtRatio": 0.999,
    "NumberOfOpenCreditLinesAndLoans": 0.999,
    "NumberRealEstateLoansOrLines": 0.999,
    "NumberOfTime30-59DaysPastDueNotWorse": 0.999,
    "NumberOfTimes90DaysLate": 0.999,
    "NumberOfTime60-89DaysPastDueNotWorse": 0.999,
}

AGE_MIN, AGE_MAX = 18, 100


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
    return df


class PreprocessTransformer(BaseEstimator, TransformerMixin):
    """Cleans raw GMSC rows and produces the engineered feature matrix.

    Attributes set in ``fit``:
        caps_                 dict: column -> winsorization cap
        income_med_by_age_    dict: age decade -> median log1p income
        income_global_med_    float: fallback median log1p income
        dependents_med_       float: median number of dependents
        past_due_med_         dict: past-due column -> median (sentinel impute)
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y=None):
        X = self._as_df(X)
        caps = {
            col: float(np.nanquantile(X[col].to_numpy(dtype=float), q))
            for col, q in WINSORIZE_SPEC.items()
        }
        # per-age-decade median of log-income (only rows with income > 0)
        inc = X[["age", "MonthlyIncome"]].copy()
        inc["log_income"] = np.log1p(inc["MonthlyIncome"].clip(lower=0).to_numpy(dtype=float))
        valid = inc[inc["MonthlyIncome"] > 0]
        valid["decade"] = np.clip(valid["age"].to_numpy(dtype=float) // 10, 1, 10).astype(int)
        income_med_by_age = valid.groupby("decade")["log_income"].median()
        if income_med_by_age.empty:
            income_med_by_age = pd.Series([np.log1p(5000.0)], index=[5])
        past_due_med = {
            c: float(np.nanmedian(X[c].to_numpy(dtype=float))) for c in PAST_DUE_COLS
        }
        self.caps_ = caps
        self.income_med_by_age_ = income_med_by_age
        self.income_global_med_ = float(income_med_by_age.median())
        self.dependents_med_ = float(np.nanmedian(X["NumberOfDependents"].to_numpy(dtype=float)))
        self.past_due_med_ = past_due_med
        self.feature_names_in_ = list(X.columns)
        return self

    # ------------------------------------------------------------- transform
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = self._as_df(X)
        n = len(X)
        cols: dict[str, np.ndarray] = {}

        # --- 1. age -------------------------------------------------------
        raw_age = X["age"].to_numpy(dtype=float)
        cols["age"] = np.clip(raw_age, AGE_MIN, AGE_MAX)
        cols["age_outlier_flag"] = ((raw_age < AGE_MIN) | (raw_age > AGE_MAX)).astype(float)
        cols["age_decade"] = (cols["age"] // 10).astype(float)

        # --- 2. past-due sentinels (96/98) -> missing + flag ---------------
        cols["pd_sentinel_flag"] = np.zeros(n, dtype=float)
        for c in PAST_DUE_COLS:
            v = X[c].to_numpy(dtype=float)
            is_sentinel = np.isin(v, PAST_DUE_SENTINELS)
            v = np.where(is_sentinel, self.past_due_med_[c], v)
            cols["pd_sentinel_flag"] += is_sentinel.astype(float)
            cols[c] = v
        cols["pd_sentinel_flag"] = (cols["pd_sentinel_flag"] > 0).astype(float)

        # --- 3. winsorize outliers ----------------------------------------
        for col, cap in self.caps_.items():
            cols[col] = np.minimum(X[col].to_numpy(dtype=float), cap)

        # --- 4. income: flag + log + impute by age decade ------------------
        raw_income = X["MonthlyIncome"].to_numpy(dtype=float)
        income_missing = np.isnan(raw_income) | (raw_income <= 0)
        log_income = np.log1p(np.maximum(raw_income, 0.0))
        decade_idx = np.clip(cols["age"] // 10, 1, 10).astype(int)
        med_arr = np.array(
            [self.income_med_by_age_.get(d, self.income_global_med_) for d in decade_idx]
        )
        cols["MonthlyIncome_log"] = np.where(income_missing, med_arr, log_income)
        cols["income_missing_flag"] = income_missing.astype(float)

        # --- 5. dependents: flag + median impute ---------------------------
        raw_dep = X["NumberOfDependents"].to_numpy(dtype=float)
        dep_missing = np.isnan(raw_dep)
        cols["NumberOfDependents"] = np.where(dep_missing, self.dependents_med_, raw_dep)
        cols["dependents_missing_flag"] = dep_missing.astype(float)

        # --- 6. engineered features ----------------------------------------
        util = cols["RevolvingUtilizationOfUnsecuredLines"]
        debt = cols["DebtRatio"]
        inc = cols["MonthlyIncome_log"]
        dep = cols["NumberOfDependents"]
        age = cols["age"]
        open_lines = cols["NumberOfOpenCreditLinesAndLoans"]
        real_estate = cols["NumberRealEstateLoansOrLines"]
        pd30 = cols["NumberOfTime30-59DaysPastDueNotWorse"]
        pd90 = cols["NumberOfTimes90DaysLate"]
        pd6089 = cols["NumberOfTime60-89DaysPastDueNotWorse"]

        total_del = pd30 + pd90 + pd6089
        max_del = np.maximum(np.maximum(pd30, pd90), pd6089)
        cols["total_delinquency"] = total_del
        cols["max_delinquency"] = max_del
        cols["delinquency_rate"] = total_del / (open_lines + 1.0)
        cols["any_delinquency_flag"] = (total_del > 0).astype(float)

        cols["utilization_bucket"] = np.digitize(util, bins=[0.1, 0.3, 0.5, 0.8, 1.0])
        cols["high_utilization_flag"] = (util > 1.0).astype(float)
        cols["utilization_x_income"] = util * inc
        cols["utilization_x_debt"] = util * debt

        income_raw = np.expm1(inc)
        cols["income_per_dependent"] = np.log1p(income_raw / (dep + 1.0))
        cols["income_to_age"] = income_raw / (age + 1e-9)
        cols["real_estate_share"] = real_estate / (open_lines + 1.0)
        cols["debt_income_ratio_flag"] = (debt > 1.0).astype(float)

        # enforce stable column order
        names = sorted(cols.keys())
        return np.column_stack([cols[k] for k in names])

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _as_df(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return sorted(
            set(self.caps_.keys())
            | {
                "age", "age_outlier_flag", "age_decade", "pd_sentinel_flag",
                "MonthlyIncome_log", "income_missing_flag", "dependents_missing_flag",
                "NumberOfDependents", "total_delinquency", "max_delinquency",
                "delinquency_rate", "any_delinquency_flag", "utilization_bucket",
                "high_utilization_flag", "utilization_x_income", "utilization_x_debt",
                "income_per_dependent", "income_to_age", "real_estate_share",
                "debt_income_ratio_flag",
            }
        )


FEATURE_GROUPS = {
    "credit utilization": ["RevolvingUtilizationOfUnsecuredLines", "utilization_bucket",
                           "high_utilization_flag", "utilization_x_income", "utilization_x_debt"],
    "debt burden": ["DebtRatio", "debt_income_ratio_flag", "income_to_age"],
    "income": ["MonthlyIncome_log", "income_missing_flag", "income_per_dependent"],
    "delinquency history": ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTimes90DaysLate",
                            "NumberOfTime60-89DaysPastDueNotWorse", "total_delinquency",
                            "max_delinquency", "delinquency_rate", "any_delinquency_flag",
                            "pd_sentinel_flag"],
    "accounts / credit lines": ["NumberOfOpenCreditLinesAndLoans", "NumberRealEstateLoansOrLines",
                                "real_estate_share"],
    "demographics": ["age", "age_decade", "age_outlier_flag", "NumberOfDependents",
                     "dependents_missing_flag"],
}
