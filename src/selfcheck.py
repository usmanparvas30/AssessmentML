"""Runnable self-check for the pieces that would fail silently.

Not a test suite - just the smallest set of assertions that break if the
leak-free encoder, the December feature path or the dollar conversion regress.

    python src/selfcheck.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import data as D
import features as F
import models as M
import pipeline as P


def check_december_path(frames: dict) -> None:
    """The December rows ship without coordinates, market index or quote signal.
    They must still reach the model with every feature populated."""
    december = frames["december"]
    matrix = F.design_matrix(december, F.LaneEncoder().fit(
        frames["train"], P.log_rpm(frames["train"])).transform(december))
    assert len(december) == 31, december.shape
    missing = [c for c in F.FEATURE_COLUMNS if matrix[c].isna().any()]
    assert not missing, f"December rows have unfilled features: {missing}"
    assert december["market_daily"].notna().all(), "December market calendar has holes"
    assert (december["pickup_lat"].notna() & december["delivery_lat"].notna()).all()
    # The chart must move with the date, otherwise the calendar features are dead.
    assert december["days_to_holiday"].nunique() > 1


def check_encoder_is_leak_free(frames: dict) -> None:
    """Out-of-fold encodings must not reproduce a row's own contribution."""
    sample = frames["train"].sample(4000, random_state=0).reset_index(drop=True)
    target = P.log_rpm(sample)
    encoder = F.LaneEncoder()
    oof = encoder.fit_transform_oof(sample, target, seed=0)
    in_fold = encoder.transform(sample)
    assert not np.allclose(oof["te_lane"], in_fold["te_lane"]), "OOF encoding equals in-fold encoding"
    assert abs(oof["te_lane"].corr(target)) < abs(in_fold["te_lane"].corr(target))


def check_unseen_cities_fall_back(frames: dict) -> None:
    """Eight validation cities never appear in training; they must encode to the
    global level and be flagged rather than producing NaN."""
    encoder = F.LaneEncoder().fit(frames["train"], P.log_rpm(frames["train"]))
    encoded = encoder.transform(frames["validation"])
    assert encoded.notna().all().all(), "unseen city produced a NaN encoding"
    unseen = encoded["seen_origin"] == 0
    assert unseen.any(), "expected some unseen origins in validation"
    assert np.allclose(encoded.loc[unseen, "te_origin"], encoder.global_level)


def check_dollar_conversion() -> None:
    log_rpm = np.log(np.array([2.0, 3.0]))
    distance = np.array([100.0, 200.0])
    assert np.allclose(M.to_dollars(log_rpm, distance), [200.0, 600.0])
    assert (M.to_dollars(np.array([-50.0]), np.array([1.0])) > 0).all(), "rates must stay positive"


def check_ensemble_is_unfitted() -> None:
    """The shipped ensemble must not depend on held-out data in any way.

    Guards the regression that motivated it: fitted weights let a learner that
    merely looked good on the blend window dominate, and cost $250 of MAE on the
    earliest fold.
    """
    members = list(M.SHIPPED_MEMBERS)
    weights = M.ensemble_weights(members + ["ridge", "forest"])
    assert list(weights.index) == members, weights.to_dict()
    assert np.allclose(weights.values, 1.0 / len(members)), "shipped weights are not equal"
    assert np.isclose(weights.sum(), 1.0)
    # Same answer whatever data is around: nothing here is fitted.
    assert weights.equals(M.ensemble_weights(members))


def check_cleaning(frames: dict) -> None:
    train = frames["train"]
    assert (train["weight"].dropna() >= 0).all(), "negative weights survived cleaning"
    assert train["market_index"].notna().all(), "market_index still has holes after fill"
    assert "quote_signal" not in train.columns, "the decoy column reached the model"


def main() -> None:
    frames = P.prepare()
    checks = [
        ("cleaning", lambda: check_cleaning(frames)),
        ("december feature path", lambda: check_december_path(frames)),
        ("encoder is leak-free", lambda: check_encoder_is_leak_free(frames)),
        ("unseen cities fall back", lambda: check_unseen_cities_fall_back(frames)),
        ("dollar conversion", check_dollar_conversion),
        ("ensemble is unfitted", check_ensemble_is_unfitted),
    ]
    for name, check in checks:
        check()
        print(f"ok  {name}")
    print(f"\n{len(checks)} checks passed")


if __name__ == "__main__":
    main()
