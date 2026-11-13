"""
Retrain the Histogram Gradient Boosting model locally.

Why: best_model.joblib was pickled in Google Colab (scikit-learn 1.6.1). Pickles
are only guaranteed to load with the same scikit-learn version, and Python 3.14
cannot install 1.6.1. Retraining locally produces a model that matches the
scikit-learn version installed on THIS machine.

What is preserved:
  * same dataset (hourly_consumption.csv)
  * same 12 features and chronological 80:20 split (see features.py)
  * same hyper-parameters, read from the original Colab model
  * model_comparison_results.csv and actual_vs_predicted.csv are NOT touched
    (they are the original Colab results and are never overwritten).

Run:  python train_model.py
"""
from __future__ import annotations

import json
import platform
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import FEATURES, TARGET, TRAIN_FRACTION, chronological_split, load_series, make_features

BASE = Path(__file__).parent
DATA = BASE / "hourly_consumption.csv"
MODEL_PATH = BASE / "best_model.joblib"
MODELS_DIR = BASE / "models"
BACKUP = MODELS_DIR / "best_model_colab_original.joblib"
META = MODELS_DIR / "model_metadata.json"
LOCAL_PREDS = MODELS_DIR / "local_actual_vs_predicted.csv"

# Hyper-parameters recovered from the original Colab model (best_model.joblib).
# random_state=42 keeps the early-stopping validation split reproducible.
HGB_PARAMS = dict(
    learning_rate=0.08,
    max_iter=150,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    random_state=42,
)

# Metrics reported for the original Colab run (for comparison only).
ORIGINAL = {"MAE": 0.3295, "RMSE": 0.4760, "R2": 0.5882}


def main() -> None:
    if not DATA.exists():
        sys.exit(f"ERROR: {DATA.name} not found next to train_model.py")
    MODELS_DIR.mkdir(exist_ok=True)

    # Keep the untouched Colab pickle (it cannot be loaded on newer scikit-learn,
    # but we never destroy the original artifact).
    if MODEL_PATH.exists() and not BACKUP.exists():
        shutil.copy2(MODEL_PATH, BACKUP)
        print(f"Backed up original Colab model -> {BACKUP.relative_to(BASE)}")

    series = load_series(DATA)
    data = make_features(series)
    train, test = chronological_split(data)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    # --- sanity checks: chronological split, no overlap, no leakage ---------
    assert train.index.max() < test.index.min(), "Train/test overlap: split is not chronological"
    assert list(X_train.columns) == FEATURES and X_train.shape[1] == 12
    assert TARGET not in X_train.columns
    print(f"Series rows        : {len(series):,}  ({series.index.min()} -> {series.index.max()})")
    print(f"Supervised rows    : {len(data):,}  (first 168 dropped for lag_168 / rolling_mean_168)")
    print(f"Train / test       : {len(train):,} / {len(test):,}  (chronological {TRAIN_FRACTION:.0%}:{1 - TRAIN_FRACTION:.0%})")
    print(f"Train period       : {train.index.min()} -> {train.index.max()}")
    print(f"Test period        : {test.index.min()} -> {test.index.max()}")

    model = HistGradientBoostingRegressor(**HGB_PARAMS)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mse = mean_squared_error(y_test, pred)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_test, pred)),
    }
    print("\nLocal retrain results (test set):")
    for k in ("MAE", "RMSE", "R2"):
        print(f"  {k:<5}: {metrics[k]:.4f}   (original Colab: {ORIGINAL[k]:.4f})")

    joblib.dump(model, MODEL_PATH)
    pd.DataFrame({"Actual": y_test, "Predicted": pred}).rename_axis("datetime").to_csv(LOCAL_PREDS)

    # Round-trip check: the file we just wrote must load in this environment.
    reloaded = joblib.load(MODEL_PATH)
    assert np.allclose(reloaded.predict(X_test), pred), "Reloaded model disagrees with trained model"

    meta = {
        "model": "HistGradientBoostingRegressor",
        "features": FEATURES,
        "hyperparameters": HGB_PARAMS,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_end": str(train.index.max()),
        "test_start": str(test.index.min()),
        "test_end": str(test.index.max()),
        "local_test_metrics": metrics,
        "original_colab_metrics": ORIGINAL,
        "early_stopping_iterations_used": int(model.n_iter_),
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    }
    META.write_text(json.dumps(meta, indent=2))
    print(f"\nSaved model    -> {MODEL_PATH.name}")
    print(f"Saved metadata -> {META.relative_to(BASE)}")
    print("Done. Launch the dashboard with:  python -m streamlit run dashboard.py")


if __name__ == "__main__":
    main()