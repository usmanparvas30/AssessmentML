"""Model zoo, blending and the log-rate-per-mile <-> dollars conversion."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def model_zoo(seed: int = 0) -> dict:
    """Four learners with genuinely different failure modes.

    `hgb_mae` is the workhorse: roughly 1% of rows carry an injected
    multiplicative shock (the log residual tails are mirrored at -1.07/+1.06), and
    an absolute-error objective estimates the conditional median, which those
    shocks cannot drag around the way a squared-error mean can.
    """
    return {
        "hgb_mae": HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.06, max_iter=700,
            max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=False, random_state=seed,
        ),
        "hgb_mse": HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.06, max_iter=600,
            max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=False, random_state=seed + 1,
        ),
        "hgb_deep": HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.04, max_iter=900,
            max_leaf_nodes=127, min_samples_leaf=20, l2_regularization=2.0,
            early_stopping=False, random_state=seed + 2,
        ),
        "forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=350, min_samples_leaf=4, max_features=0.4,
                n_jobs=-1, random_state=seed + 3,
            ),
        ),
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=3.0)
        ),
    }


def to_dollars(log_rpm, distance, calibration: float = 1.0) -> np.ndarray:
    """log(rate per mile) back to a positive dollar rate."""
    return np.clip(np.exp(np.asarray(log_rpm)) * np.asarray(distance) * calibration, 1.0, None)


# The shipped model is a fixed, equal-weight average of the two absolute-error
# boosters. It is deliberately *not* a fitted blend - see `ensemble_weights`.
SHIPPED_MEMBERS = ("hgb_mae", "hgb_deep")


def ensemble_weights(names) -> pd.Series:
    """Equal weights over the shipped members. No parameters are fitted here.

    Three ensembling schemes were tried and measured under rolling-origin
    validation before settling on none of them:

    1. Non-negative least squares in log space produced a blend that scored worse
       than its own best member - log-space least squares is not the graded metric.
    2. Greedy selection against dollar MAE fixed that, then overfitted the blend
       window: on the earliest fold it gave the ridge model 74% of the weight, and
       ridge went on to score $498 MAE on the block it was meant to predict.
    3. Bagging the greedy selection changed the weights by less than a percentage
       point, which is what ruled out sampling noise as the explanation.

    The actual cause is a mismatch that no weighting scheme can fix: weights are
    learned from base models fitted on the pre-blend window, then applied to models
    refitted on that window plus the blend window. On the earliest fold that refit
    doubles the training data, the boosters improve sharply, ridge barely moves,
    and the weights are stale before they are ever used.

    Across all three folds fitted blending never beat the best single member - it
    either tied or lost badly - so the fitting is gone. Averaging the two
    absolute-error boosters is a free variance reduction with nothing fitted, and
    it avoids picking between two learners separated by $0.10 of cross-validated
    MAE.
    """
    members = [name for name in names if name in SHIPPED_MEMBERS]
    if not members:
        raise ValueError(f"none of {SHIPPED_MEMBERS} present in {list(names)}")
    return pd.Series(1.0 / len(members), index=members)


def calibration_factor(log_rpm_pred, distance, actual_rate, grid=None) -> float:
    """One multiplicative factor chosen on held-out data to minimise dollar MAE.

    Converting a log-space prediction back to dollars is biased; rather than
    assuming the textbook smearing correction is the right one for an absolute
    error metric, the factor is simply searched for.
    """
    grid = np.linspace(0.94, 1.06, 61) if grid is None else grid
    errors = [
        np.mean(np.abs(to_dollars(log_rpm_pred, distance, c) - actual_rate)) for c in grid
    ]
    return float(grid[int(np.argmin(errors))])


def metrics(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    ape = np.abs(error) / actual
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mape": float(np.mean(ape) * 100),
        "medape": float(np.median(ape) * 100),
        "r2": float(1 - ss_res / ss_tot),
    }
