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


def check_blend_beats_its_members() -> None:
    """Greedy selection is only worth its complexity if it cannot be worse than
    the best member on the data it is fitted on."""
    rng = np.random.default_rng(0)
    distance = rng.uniform(100, 2000, 500)
    truth = np.log(rng.uniform(1.5, 3.0, 500))
    actual = M.to_dollars(truth, distance)
    predictions = pd.DataFrame({
        "good": truth + rng.normal(0, 0.02, 500),
        "bad": truth + rng.normal(0, 0.50, 500),
    })
    weights = M.blend_weights(predictions, distance, actual)
    blended = predictions.values @ weights.values
    best_member = min(
        np.mean(np.abs(M.to_dollars(predictions[c], distance) - actual)) for c in predictions
    )
    assert np.mean(np.abs(M.to_dollars(blended, distance) - actual)) <= best_member * 1.001
    assert weights["good"] > weights["bad"], weights.to_dict()


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
        ("blend never worse than members", check_blend_beats_its_members),
    ]
    for name, check in checks:
        check()
        print(f"ok  {name}")
    print(f"\n{len(checks)} checks passed")


if __name__ == "__main__":
    main()
