"""Loading, cleaning and shared lookup tables for the freight rate model.

Every cleaning decision here is justified by `src/eda.py`; nothing is cleaned
"because it looks odd".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

TRAIN_CSV = DATA / "train_test.csv"
VALIDATION_CSV = DATA / "validation.csv"
TEMPLATE_CSV = DATA / "validation_predictions_template.csv"
DECEMBER_CSV = DATA / "december_chart_inputs.csv"

TARGET = "posted_rate"
EARTH_RADIUS_MI = 3958.8

# quote_signal is deliberately excluded everywhere: see eda.py, its correlation
# with realised rate-per-mile is 0.05. It is a decoy column, not a price signal.
DROP_COLUMNS = ["quote_signal"]


def haversine_miles(lat1, lon1, lat2, lon2):
    rad = np.pi / 180.0
    dlat = (lat2 - lat1) * rad
    dlon = (lon2 - lon1) * rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1 * rad) * np.cos(lat2 * rad) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_MI * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def initial_bearing(lat1, lon1, lat2, lon2):
    rad = np.pi / 180.0
    dlon = (lon2 - lon1) * rad
    y = np.sin(dlon) * np.cos(lat2 * rad)
    x = np.cos(lat1 * rad) * np.sin(lat2 * rad) - np.sin(lat1 * rad) * np.cos(lat2 * rad) * np.cos(dlon)
    return np.arctan2(y, x)


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train, validation and the December chart inputs, dates parsed."""
    train = pd.read_csv(TRAIN_CSV, parse_dates=["date"])
    validation = pd.read_csv(VALIDATION_CSV, parse_dates=["date"])
    december = pd.read_csv(DECEMBER_CSV, parse_dates=["date"])
    return train, validation, december


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """One row per city. Coordinates are fixed per city in this dataset, so the
    December inputs (which ship without lat/lon) can be joined back to geography.
    """
    parts = []
    for frame in frames:
        for prefix in ("pickup", "delivery"):
            lat, lon = f"{prefix}_lat", f"{prefix}_lon"
            if lat not in frame.columns:
                continue
            part = frame[[prefix, lat, lon]].copy()
            part.columns = ["city", "lat", "lon"]
            parts.append(part)
    cities = pd.concat(parts, ignore_index=True).dropna()
    return cities.groupby("city", as_index=False)[["lat", "lon"]].median()


def market_calendar(*frames: pd.DataFrame) -> pd.DataFrame:
    """Daily market state.

    `market_index` is published per load but is really a daily market reading seen
    through per-row noise: its row-level correlation with rate-per-mile is 0.08
    while the daily-mean correlation is 0.58. Aggregating it to a calendar is what
    makes it usable, and it also gives the December chart rows (which ship without
    the column) a market feature.
    """
    parts = [f[["date", "market_index"]] for f in frames if "market_index" in f.columns]
    daily = pd.concat(parts, ignore_index=True).groupby("date", as_index=False).agg(
        market_daily=("market_index", "mean"),
        market_daily_std=("market_index", "std"),
    )
    daily = daily.sort_values("date").reset_index(drop=True)
    full = pd.DataFrame({"date": pd.date_range(daily.date.min(), daily.date.max(), freq="D")})
    daily = full.merge(daily, on="date", how="left")
    daily["market_daily"] = daily["market_daily"].interpolate().ffill().bfill()
    daily["market_daily_std"] = daily["market_daily_std"].interpolate().ffill().bfill()
    daily["market_7d"] = daily["market_daily"].rolling(7, min_periods=1).mean()
    daily["market_28d"] = daily["market_daily"].rolling(28, min_periods=1).mean()
    daily["market_trend"] = daily["market_7d"] - daily["market_28d"]
    return daily


def clean(frame: pd.DataFrame, cities: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Repair known data-quality defects and attach the shared lookups.

    Deliberately *not* repaired: `distance` values that look long next to the
    great-circle distance. That ratio is constant within a lane (std 0.02), so it
    is distortion baked into the synthetic coordinates, not per-row corruption.
    """
    out = frame.drop(columns=[c for c in DROP_COLUMNS if c in frame.columns]).copy()

    # Weight: 292 training rows carry the correct magnitude with a flipped sign.
    out["weight_missing"] = out["weight"].isna().astype("int8")
    out["weight"] = out["weight"].abs()

    # Coordinates are constant per city, so derive them rather than trusting the
    # row copy. This is what lets the December inputs share one feature path.
    coords = cities.set_index("city")
    for prefix in ("pickup", "delivery"):
        out[f"{prefix}_lat"] = out[prefix].map(coords["lat"])
        out[f"{prefix}_lon"] = out[prefix].map(coords["lon"])

    # market_index is missing on ~0.8% of rows; the daily calendar is the natural
    # fill because the column is a noisy read of exactly that daily value.
    out = out.merge(calendar, on="date", how="left")
    if "market_index" not in out.columns:
        out["market_index"] = np.nan
    out["market_index_missing"] = out["market_index"].isna().astype("int8")
    out["market_index"] = out["market_index"].fillna(out["market_daily"])

    return out


def quality_report(train: pd.DataFrame, validation: pd.DataFrame) -> dict:
    """The numbers quoted in the write-up, computed rather than remembered."""
    rpm = train[TARGET] / train["distance"]
    lane_median = np.log(rpm).groupby([train.pickup, train.delivery]).transform("median")
    residual = np.log(rpm) - lane_median
    return {
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "train_date_range": [str(train.date.min().date()), str(train.date.max().date())],
        "validation_date_range": [str(validation.date.min().date()), str(validation.date.max().date())],
        "negative_weight_rows": int((train.weight < 0).sum()),
        "missing_weight_rows": int(train.weight.isna().sum()),
        "missing_market_index_rows": int(train.market_index.isna().sum()),
        "quote_signal_corr_with_rpm": float(train.quote_signal.corr(rpm)),
        "market_index_row_corr_with_rpm": float(train.market_index.corr(rpm)),
        "market_index_daily_corr_with_rpm": float(
            train.groupby("date").market_index.mean().corr(train.groupby("date").apply(
                lambda g: (g[TARGET] / g.distance).mean(), include_groups=False))
        ),
        "cities_in_validation_absent_from_train": sorted(
            (set(validation.pickup) | set(validation.delivery)) - (set(train.pickup) | set(train.delivery))
        ),
        "contamination_p005_log_residual": float(residual.quantile(0.005)),
        "contamination_p995_log_residual": float(residual.quantile(0.995)),
    }
