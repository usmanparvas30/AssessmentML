"""Exploration: the numbers and figures quoted in the report.

Run this before trusting anything in the model - every cleaning decision and every
dropped column is justified by an output of this script.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data as D

FIGURES = D.REPORTS / "figures"
INK = "#064A56"


def figure_market_vs_rate(train: pd.DataFrame) -> None:
    """Rate per mile and the market index, both aggregated to a day.

    Row level these two columns look almost unrelated; daily they move together.
    That gap is the whole argument for the market calendar feature.
    """
    daily = train.assign(rpm=train[D.TARGET] / train["distance"]).groupby("date").agg(
        rpm=("rpm", "mean"), market=("market_index", "mean")
    )
    figure, axis = plt.subplots(figsize=(10, 4), dpi=150)
    axis.plot(daily.index, daily["rpm"], color=INK, linewidth=1.6, label="mean rate per mile ($)")
    twin = axis.twinx()
    twin.plot(daily.index, daily["market"], color="#C2703D", linewidth=1.6, alpha=0.85, label="mean market index")
    axis.set_title("Daily rate per mile tracks the market index", loc="left", fontweight="bold")
    axis.set_ylabel("$ / mile")
    twin.set_ylabel("market index")
    axis.grid(axis="y", color="#D9E2E4", linewidth=0.8)
    axis.spines[["top"]].set_visible(False)
    twin.spines[["top"]].set_visible(False)
    lines = axis.get_lines() + twin.get_lines()
    axis.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper left")
    figure.tight_layout()
    figure.savefig(FIGURES / "market_vs_rate.png", bbox_inches="tight")
    plt.close(figure)


def figure_rpm_vs_distance(train: pd.DataFrame) -> None:
    """Rate per mile decays with length of haul - the reason the model predicts
    log(rate per mile) with distance as an offset instead of raw dollars."""
    frame = train.assign(rpm=train[D.TARGET] / train["distance"])
    figure, axis = plt.subplots(figsize=(8, 4), dpi=150)
    for equipment, color in zip(["Dry Van", "Reefer", "Flatbed"], [INK, "#C2703D", "#5B8C5A"]):
        part = frame[frame.equipment == equipment]
        binned = part.groupby(pd.cut(part.distance, np.arange(0, 3600, 150)), observed=True).rpm.median()
        centres = [interval.mid for interval in binned.index]
        axis.plot(centres, binned.values, marker="o", markersize=3, color=color, label=equipment)
    axis.set_title("Rate per mile by length of haul and equipment", loc="left", fontweight="bold")
    axis.set_xlabel("distance (miles)")
    axis.set_ylabel("median $ / mile")
    axis.grid(axis="y", color="#D9E2E4", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(FIGURES / "rpm_vs_distance.png", bbox_inches="tight")
    plt.close(figure)


def figure_contamination(train: pd.DataFrame) -> None:
    """Log rate-per-mile against the lane's own median.

    The tails are mirrored, which is the signature of an injected multiplicative
    shock rather than a fat-tailed price distribution - and the reason the primary
    learner optimises absolute error.
    """
    rpm = np.log(train[D.TARGET] / train["distance"])
    residual = rpm - rpm.groupby([train.pickup, train.delivery]).transform("median")
    figure, axis = plt.subplots(figsize=(8, 4), dpi=150)
    axis.hist(residual, bins=200, color=INK, alpha=0.85)
    for quantile in (0.005, 0.995):
        axis.axvline(residual.quantile(quantile), color="#C2703D", linestyle="--", linewidth=1.2)
    axis.set_yscale("log")
    axis.set_title("Log rate-per-mile vs lane median: mirrored contamination tails",
                   loc="left", fontweight="bold")
    axis.set_xlabel("log residual")
    axis.set_ylabel("rows (log scale)")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(FIGURES / "contamination.png", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    train, validation, _ = D.load_raw()
    FIGURES.mkdir(parents=True, exist_ok=True)

    report = D.quality_report(train, validation)
    rpm = train[D.TARGET] / train["distance"]
    report["rate_per_mile_quantiles"] = {
        str(q): float(rpm.quantile(q)) for q in (0.01, 0.25, 0.5, 0.75, 0.99)
    }
    report["rate_per_mile_by_equipment"] = rpm.groupby(train.equipment).median().round(4).to_dict()
    report["monthly_mean_rate_per_mile"] = {
        str(period): float(value)
        for period, value in rpm.groupby(train.date.dt.to_period("M")).mean().items()
    }
    report["distance_corr_with_rate"] = float(train.distance.corr(train[D.TARGET]))

    (D.REPORTS / "eda_report.json").write_text(json.dumps(report, indent=2))
    figure_market_vs_rate(train)
    figure_rpm_vs_distance(train)
    figure_contamination(train)

    print(json.dumps(report, indent=2))
    print(f"\nFigures written to {FIGURES.relative_to(D.ROOT)}")


if __name__ == "__main__":
    main()
