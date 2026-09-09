"""End-to-end pipeline: raw CSVs -> cleaned frames -> fitted stack -> dollars."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import data as D
import features as F
import models as M


def prepare() -> dict:
    """Clean all three frames through one shared path.

    The market calendar and the city table are built from feature columns of every
    frame - never from labels - which is what allows the December chart rows,
    shipped without coordinates or a market column, to be scored by the same model
    as the validation set.
    """
    train, validation, december = D.load_raw()
    cities = D.city_coordinates(train, validation)
    calendar = D.market_calendar(train, validation)
    return {
        "train": F.base_features(D.clean(train, cities, calendar)),
        "validation": F.base_features(D.clean(validation, cities, calendar)),
        "december": F.base_features(D.clean(december, cities, calendar)),
        "cities": cities,
        "calendar": calendar,
        "raw_train": train,
        "raw_validation": validation,
    }


def log_rpm(frame: pd.DataFrame) -> pd.Series:
    return np.log(frame[D.TARGET] / frame["distance"])


@dataclass
class Stack:
    """Fitted base learners plus the blend weights and dollar calibration."""

    encoder: F.LaneEncoder
    fitted: dict
    weights: pd.Series
    calibration: float = 1.0
    holdout_report: dict = field(default_factory=dict)

    def base_predictions(self, frame: pd.DataFrame, names=None) -> pd.DataFrame:
        matrix = F.design_matrix(frame, self.encoder.transform(frame))
        names = list(self.fitted) if names is None else list(names)
        return pd.DataFrame(
            {name: self.fitted[name].predict(matrix) for name in names}, index=frame.index
        )

    def predict_log_rpm(self, frame: pd.DataFrame) -> np.ndarray:
        # Only the shipped members are run at inference; the rest exist to be
        # reported against.
        base = self.base_predictions(frame, self.weights.index)
        return base[self.weights.index].values @ self.weights.values

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return M.to_dollars(self.predict_log_rpm(frame), frame["distance"], self.calibration)


def fit_stack(fit_frame: pd.DataFrame, blend_frame: pd.DataFrame, seed: int = 0,
              refit_on_all: bool = True) -> Stack:
    """Fit base models, calibrate on `blend_frame`, then
    optionally refit the base models on the union so the shipped model has seen the
    most recent weeks - the ones closest to the November/December target period.
    """
    encoder = F.LaneEncoder()
    y_fit = log_rpm(fit_frame)
    encodings = encoder.fit_transform_oof(fit_frame, y_fit, seed=seed)
    matrix = F.design_matrix(fit_frame, encodings)

    fitted = {name: model.fit(matrix, y_fit) for name, model in M.model_zoo(seed).items()}
    # Every learner is fitted so the comparison table in the report is real, but
    # only the shipped members carry weight in the prediction.
    stage = Stack(encoder=encoder, fitted=fitted, weights=M.ensemble_weights(fitted))

    blend_base = stage.base_predictions(blend_frame)
    weights = M.ensemble_weights(blend_base.columns)
    blended = blend_base[weights.index].values @ weights.values
    calibration = M.calibration_factor(
        blended, blend_frame["distance"].values, blend_frame[D.TARGET].values
    )

    report = {
        name: M.metrics(blend_frame[D.TARGET], M.to_dollars(blend_base[name], blend_frame["distance"]))
        for name in blend_base.columns
    }
    report["blend"] = M.metrics(
        blend_frame[D.TARGET], M.to_dollars(blended, blend_frame["distance"], calibration)
    )

    if refit_on_all:
        combined = pd.concat([fit_frame, blend_frame], ignore_index=True)
        encoder = F.LaneEncoder()
        y_all = log_rpm(combined)
        encodings = encoder.fit_transform_oof(combined, y_all, seed=seed)
        matrix = F.design_matrix(combined, encodings)
        fitted = {name: model.fit(matrix, y_all) for name, model in M.model_zoo(seed).items()}

    return Stack(encoder=encoder, fitted=fitted, weights=weights,
                 calibration=calibration, holdout_report=report)
