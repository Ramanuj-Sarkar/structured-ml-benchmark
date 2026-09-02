"""SHAP-based explainability for the trained GBDT models.

Loads the fitted ``PreprocessTransformer`` + models saved by ``train.py``,
computes TreeExplainer SHAP values on a seeded subsample of the untouched
test set (no leakage), and writes:

* ``figures/shap_summary_{model}.png``   — beeswarm summary (feature impact)
* ``figures/shap_importance_{model}.png`` — mean |SHAP| bar chart
* ``figures/shap_dependence_{feat}.png``  — dependence plots for the top-2
  features of the best model (colored by the second-most-important feature)
* ``figures/shap_model_comparison.png``   — mean |SHAP| across the 3 models
* ``results/shap_summary.json``           — machine-readable importances
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import shap
from data_prep import PreprocessTransformer, TARGET, load_data
from train import SEED, TEST_SIZE, BUILDERS, MODELS_DIR, RESULTS_DIR, FIGURES_DIR

SHAP_SAMPLE = 2000


def load_artifacts():
    df = load_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X_test), size=min(SHAP_SAMPLE, len(X_test)), replace=False)
    X_sample_raw = X_test.iloc[idx]
    y_sample = y_test.iloc[idx]

    artifacts = {}
    for name in BUILDERS:
        art = joblib.load(MODELS_DIR / f"{name}.joblib")
        prep: PreprocessTransformer = art["prep"]
        model = art["model"]
        X_p = prep.transform(X_sample_raw)
        X_df = pd.DataFrame(X_p, columns=prep.get_feature_names_out())
        artifacts[name] = {"model": model, "prep": prep, "X_df": X_df,
                           "y": y_sample.to_numpy()}
    return artifacts, X_test


def explain_model(name, art, out) -> dict:
    print(f"  [{name}] computing SHAP on {len(art['X_df'])} test rows ...")
    explainer = shap.TreeExplainer(art["model"])
    sv = explainer(art["X_df"])  # Explanation object
    mean_abs = np.abs(sv.values).mean(axis=0)
    order = np.argsort(-mean_abs)
    feats = art["X_df"].columns

    # beeswarm summary
    plt.figure(figsize=(9, 6.5))
    shap.plots.beeswarm(sv, max_display=18, show=False)
    plt.title(f"SHAP summary — {name} (test subsample, n={len(art['X_df'])})")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"shap_summary_{name}.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # mean |SHAP| bar
    plt.figure(figsize=(8, 5.5))
    shap.plots.bar(sv, max_display=18, show=False)
    plt.title(f"Mean |SHAP| — {name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"shap_importance_{name}.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # store machine-readable importance
    importance = {
        str(feats[i]): float(mean_abs[i]) for i in order
    }
    return {"mean_abs_shap": importance,
            "top_features": [str(feats[i]) for i in order[:10]]}


def dependence_plots(name, art, top2, out):
    print(f"  [{name}] dependence plots for {top2}")
    explainer = shap.TreeExplainer(art["model"])
    sv = explainer(art["X_df"])
    f0, f1 = top2[0], top2[1]
    for feat in (f0, f1):
        other = f1 if feat == f0 else f0
        plt.figure(figsize=(7, 5))
        shap.plots.scatter(sv[:, feat], color=sv[:, other], show=False)
        plt.title(f"SHAP dependence — {feat} (colored by {other})")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"shap_dependence_{feat}.png", dpi=150,
                    bbox_inches="tight")
        plt.close("all")


def model_comparison(importances: dict[str, dict]):
    df = pd.DataFrame(importances).fillna(0.0)
    df["max"] = df.max(axis=1)
    top = df.sort_values("max", ascending=False).head(15).drop(columns=["max"])
    fig, ax = plt.subplots(figsize=(9, 6.5))
    y = np.arange(len(top))
    height = 0.27
    colors = {"xgb": "#4C72B0", "lgbm": "#DD8452", "cat": "#55A868"}
    for i, col in enumerate(importances):
        ax.barh(y + (i - 1) * height, top[col].values, height=height,
                label=col, color=colors[col], alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(top.index)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP| (test subsample)")
    ax.set_title("Feature importance across models (mean |SHAP|)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_model_comparison.png", dpi=150)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    artifacts, _ = load_artifacts()

    print("Computing SHAP explanations (test subsample)...")
    importances = {}
    for name, art in artifacts.items():
        res = explain_model(name, art, RESULTS_DIR)
        importances[name] = res["mean_abs_shap"]
        print(f"  top features ({name}): {res['top_features'][:5]}")

    # dependence plots for the best model by test AUC
    with open(RESULTS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    best_model = max(BUILDERS, key=lambda m: metrics["test"][m]["auc"])
    top2 = importances[best_model]
    top2 = sorted(top2, key=lambda k: -top2[k])[:2]
    print(f"\nBest model: {best_model}; dependence plots for {top2}")
    dependence_plots(best_model, artifacts[best_model], top2, RESULTS_DIR)

    model_comparison(importances)

    with open(RESULTS_DIR / "shap_summary.json", "w") as f:
        json.dump({"sample_size": SHAP_SAMPLE,
                   "best_model": best_model,
                   "mean_abs_shap": importances}, f, indent=2)
    print(f"Saved SHAP figures -> {FIGURES_DIR}, summary -> "
          f"{RESULTS_DIR / 'shap_summary.json'}")


if __name__ == "__main__":
    main()
