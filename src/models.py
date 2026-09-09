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


def blend_weights(predictions: pd.DataFrame, distance, actual, n_iter: int = 30) -> pd.Series:
    """Greedy ensemble selection (Caruana) on held-out data, scored in dollars.

    An earlier version fitted non-negative least squares in log space and produced
    a blend that was *worse* than its own best member: least squares in log space
    is not the objective anyone is graded on. This picks members one at a time,
    with replacement, against dollar MAE - the metric that actually matters - so a
    learner only enters the blend if it improves that number.
    """
    distance = np.asarray(distance, dtype=float)
    actual = np.asarray(actual, dtype=float)
    columns = list(predictions.columns)
    values = {name: predictions[name].to_numpy(dtype=float) for name in columns}

    def score(mean_log_rpm):
        return float(np.mean(np.abs(to_dollars(mean_log_rpm, distance) - actual)))

    counts = {name: 0 for name in columns}
    running = np.zeros(len(actual))
    chosen = 0
    best_overall, best_counts = np.inf, None
    for _ in range(n_iter):
        candidate, candidate_score = None, np.inf
        for name in columns:
            trial = (running + values[name]) / (chosen + 1)
            trial_score = score(trial)
            if trial_score < candidate_score:
                candidate, candidate_score = name, trial_score
        running = running + values[candidate]
        chosen += 1
        counts[candidate] += 1
        if candidate_score < best_overall:
            best_overall, best_counts = candidate_score, dict(counts)

    weights = pd.Series(best_counts, index=columns, dtype=float)
    return weights / weights.sum()


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
