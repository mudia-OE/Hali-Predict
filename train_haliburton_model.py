"""Estimate Tyrese Haliburton's post-Achilles per-36 production.

This is deliberately a small, interpretable model.  With only 23 historical
cases, a large model (random forest, boosting, neural net) will mostly memorize
the dataset.  We model each player's change in per-36 stats with ridge
regression, then add that change to Haliburton's pre-injury numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGETS = ["PostPTS36", "PostAST36", "PostREB36"]
PRE_TARGETS = ["PrePTS36", "PreAST36", "PreREB36"]
NUMERIC_FEATURES = ["Age", "Height", "Weight", "RecoveryMonths", *PRE_TARGETS]
CATEGORICAL_FEATURES = ["Position"]
FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


def make_model() -> Pipeline:
    """Use one model per target internally; Ridge supports multi-output labels."""
    preprocess = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("position", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    # Fixed regularization is more stable than tuning hyperparameters on 23 rows.
    return Pipeline([( "features", preprocess), ("ridge", Ridge(alpha=20.0))])


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8-sig")
    required = {"Player", *FEATURES, *TARGETS}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    if data[list(required)].isna().any().any():
        raise ValueError("CSV has missing values in model columns.")
    return data


def leave_one_out_predictions(data: pd.DataFrame) -> np.ndarray:
    """Out-of-sample predictions for an honest estimate of model error."""
    x = data[FEATURES]
    # Modeling the post-minus-pre change anchors estimates to player baseline.
    changes = data[TARGETS].to_numpy() - data[PRE_TARGETS].to_numpy()
    predictions = np.empty_like(changes, dtype=float)
    for train_idx, test_idx in LeaveOneOut().split(x):
        model = make_model()
        model.fit(x.iloc[train_idx], changes[train_idx])
        predictions[test_idx] = model.predict(x.iloc[test_idx])
    return predictions + data[PRE_TARGETS].to_numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path(__file__).with_name("achilles_dataset.csv"))
    # Update these inputs if you choose another pre-injury season or return date.
    parser.add_argument("--age", type=float, default=25)
    parser.add_argument("--height", type=float, default=77)
    parser.add_argument("--weight", type=float, default=185)
    parser.add_argument("--position", default="pg")
    parser.add_argument("--recovery-months", type=float, default=14)
    parser.add_argument("--pre-pts36", type=float, default=25.3)
    parser.add_argument("--pre-ast36", type=float, default=10.2)
    parser.add_argument("--pre-reb36", type=float, default=3.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_data(args.csv)
    loo_post = leave_one_out_predictions(data)
    actual_post = data[TARGETS].to_numpy()
    mae = mean_absolute_error(actual_post, loo_post, multioutput="raw_values")

    # A simple 80% empirical prediction band from held-out residuals.
    abs_errors = np.abs(actual_post - loo_post)
    error_band = np.quantile(abs_errors, 0.80, axis=0)

    haliburton = pd.DataFrame(
        [{
            "Age": args.age, "Height": args.height, "Weight": args.weight,
            "Position": args.position.lower(), "RecoveryMonths": args.recovery_months,
            "PrePTS36": args.pre_pts36, "PreAST36": args.pre_ast36,
            "PreREB36": args.pre_reb36,
        }]
    )
    changes = data[TARGETS].to_numpy() - data[PRE_TARGETS].to_numpy()
    model = make_model().fit(data[FEATURES], changes)
    predicted_post = model.predict(haliburton)[0] + haliburton[PRE_TARGETS].to_numpy()[0]

    labels = ["PTS", "AST", "REB"]
    print(f"Historical cases: {len(data)}")
    print("Leave-one-out MAE (per 36): " + ", ".join(
        f"{label} {value:.2f}" for label, value in zip(labels, mae)
    ))
    print("\nHaliburton projected post-return averages (per 36):")
    for label, estimate, band in zip(labels, predicted_post, error_band):
        print(f"  {label}: {estimate:.1f}  (illustrative 80% range: {estimate-band:.1f} to {estimate+band:.1f})")


if __name__ == "__main__":
    main()
