"""End-to-end pipeline: preprocess -> tune -> CV -> test eval -> SHAP.

Run with:  python -m src.pipeline
"""

from __future__ import annotations

import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")  # keep logs readable (e.g. LGBM eval_set deprecation)

# Make sibling modules (data_prep, train, explain) importable whether this
# package is run as `python -m src.pipeline` or `python src/pipeline.py`.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

START = time.time()


def main():
    import train
    import explain

    print("=" * 70)
    print("Step 1/2: feature engineering, tuning, CV, test evaluation")
    print("=" * 70)
    train.main()

    print("\n" + "=" * 70)
    print("Step 2/2: SHAP explainability")
    print("=" * 70)
    explain.main()

    print(f"\nTotal time: {(time.time() - START) / 60:.1f} min")


if __name__ == "__main__":
    main()
