"""Feature engineering.

Two design decisions drive this module:

1. The model predicts log(rate per mile), not the rate. Distance explains 91% of
   raw rate variance, so folding it into the target as an offset leaves the model
   free to learn the part that is actually hard - how price per mile moves with
   lane, equipment, market and calendar.

2. No raw time index is used as a feature. Training stops on 2025-10-31 and the
   task is to predict November and December, and a tree cannot extrapolate past
   the largest split point it ever saw. Seasonality therefore enters only through
   cyclical calendar terms and the market calendar, both of which are defined for
   future dates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from data import haversine_miles, initial_bearing

EQUIPMENT = ["Dry Van", "Reefer", "Flatbed"]

# US federal holidays 2025 plus the two shutdown days that bracket Christmas and
# New Year. The December chart lives inside this window, so it matters.
HOLIDAYS_2025 = pd.to_datetime([
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26", "2025-06-19",
    "2025-07-04", "2025-09-01", "2025-10-13", "2025-11-11", "2025-11-27",
    "2025-11-28", "2025-12-24", "2025-12-25", "2025-12-31",
])

DISTANCE_BANDS = [0, 300, 600, 1000, 1500, 2500, 10000]


def add_geo(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame
    out["haversine"] = haversine_miles(
        out.pickup_lat, out.pickup_lon, out.delivery_lat, out.delivery_lon
    )
    # Constant within a lane, so this reads as a corridor identifier rather than
    # a data-quality flag.
    out["detour_ratio"] = out["distance"] / out["haversine"].clip(lower=1.0)
    bearing = initial_bearing(out.pickup_lat, out.pickup_lon, out.delivery_lat, out.delivery_lon)
    out["bearing_sin"] = np.sin(bearing)
    out["bearing_cos"] = np.cos(bearing)
    out["mid_lat"] = (out.pickup_lat + out.delivery_lat) / 2
    out["mid_lon"] = (out.pickup_lon + out.delivery_lon) / 2
    out["d_lat"] = out.delivery_lat - out.pickup_lat
    out["d_lon"] = out.delivery_lon - out.pickup_lon
    out["log_distance"] = np.log(out["distance"])
    return out


def add_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame
    date = out["date"]
    doy = date.dt.dayofyear
    out["day_of_week"] = date.dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("int8")
    out["day_of_month"] = date.dt.day
    out["month"] = date.dt.month
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    gap = np.abs(date.values[:, None] - HOLIDAYS_2025.values[None, :]).astype("timedelta64[D]").astype(int)
    out["days_to_holiday"] = gap.min(axis=1)
    out["is_holiday"] = (out["days_to_holiday"] == 0).astype("int8")
    out["is_holiday_week"] = (out["days_to_holiday"] <= 3).astype("int8")
    return out


def add_load(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame
    out["weight_per_mile"] = out["weight"] / out["distance"]
    out["weight_k"] = out["weight"] / 1000.0
    for equipment in EQUIPMENT:
        out[f"eq_{equipment.replace(' ', '_').lower()}"] = (out["equipment"] == equipment).astype("int8")
    out["distance_band"] = pd.cut(out["distance"], DISTANCE_BANDS, labels=False).astype("int8")
    return out


class LaneEncoder:
    """Shrunk median log(rate per mile) for lane, origin, destination and
    equipment-by-distance-band.

    Eight cities appear in the validation set that never appear in training, so
    every encoding falls back to the global level and ships a `_seen` flag; the
    model can lean on raw geography for those rows instead.
    """

    SPECS = {
        "lane": ["pickup", "delivery"],
        "origin": ["pickup"],
        "destination": ["delivery"],
        "corridor": ["pickup", "delivery", "equipment"],
        "equip_band": ["equipment", "distance_band"],
        "origin_equip": ["pickup", "equipment"],
    }

    def __init__(self, smoothing: float = 20.0):
        self.smoothing = smoothing
        self.tables: dict[str, pd.DataFrame] = {}
        self.global_level = 0.0

    def fit(self, frame: pd.DataFrame, target_log_rpm: pd.Series) -> "LaneEncoder":
        work = frame.copy()
        work["_y"] = np.asarray(target_log_rpm)
        self.global_level = float(work["_y"].median())
        for name, keys in self.SPECS.items():
            grouped = work.groupby(keys)["_y"].agg(["median", "size"])
            weight = grouped["size"] / (grouped["size"] + self.smoothing)
            grouped["value"] = weight * grouped["median"] + (1 - weight) * self.global_level
            self.tables[name] = grouped[["value", "size"]]
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        for name, keys in self.SPECS.items():
            table = self.tables[name]
            index = pd.MultiIndex.from_frame(frame[keys]) if len(keys) > 1 else pd.Index(frame[keys[0]])
            joined = table.reindex(index)
            out[f"te_{name}"] = np.where(
                joined["value"].isna(), self.global_level, joined["value"]
            )
            out[f"n_{name}"] = np.log1p(np.nan_to_num(joined["size"].to_numpy(), nan=0.0))
            out[f"seen_{name}"] = (~joined["value"].isna().to_numpy()).astype("int8")
        return out

    def fit_transform_oof(self, frame, target_log_rpm, n_splits=5, seed=0) -> pd.DataFrame:
        """Out-of-fold encodings for the rows the model is trained on, so a lane's
        own labels never reach its own features."""
        encoded = pd.DataFrame(index=frame.index, dtype=float)
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        target = pd.Series(np.asarray(target_log_rpm), index=frame.index)
        for fit_idx, apply_idx in splitter.split(frame):
            fold = LaneEncoder(self.smoothing).fit(frame.iloc[fit_idx], target.iloc[fit_idx])
            part = fold.transform(frame.iloc[apply_idx])
            encoded.loc[part.index, part.columns] = part
        self.fit(frame, target)
        return encoded[self.transform(frame.head(1)).columns]


def base_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out = add_geo(out)
    out = add_calendar(out)
    out = add_load(out)
    return out


FEATURE_COLUMNS = [
    "distance", "log_distance", "haversine", "detour_ratio",
    "bearing_sin", "bearing_cos", "mid_lat", "mid_lon", "d_lat", "d_lon",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "weight", "weight_k", "weight_per_mile", "weight_missing",
    "day_of_week", "is_weekend", "day_of_month", "month",
    "doy_sin", "doy_cos", "month_sin", "month_cos",
    "days_to_holiday", "is_holiday", "is_holiday_week",
    "market_index", "market_index_missing", "market_daily", "market_daily_std",
    "market_7d", "market_28d", "market_trend",
    "distance_band",
    "eq_dry_van", "eq_reefer", "eq_flatbed",
]


def design_matrix(frame: pd.DataFrame, encodings: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([frame[FEATURE_COLUMNS], encodings.set_index(frame.index)], axis=1)
