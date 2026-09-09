"""Rolling-origin validation.

The task is not "hold out random rows". Training data ends 2025-10-31 and the
graded predictions are November and December, so every reported number here comes
from a fold that hides a *future* two-month block, the same shape and the same
horizon as the real submission. A random split is also computed, once, purely to
show how much it flatters the model.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import data as D
import models as M
import pipeline as P

# Each fold hides a two-month future block, matching the November-December horizon.
FOLDS = [
    ("2025-05-01", "2025-06-30"),
    ("2025-07-01", "2025-08-31"),
    ("2025-09-01", "2025-10-31"),
]
BLEND_MONTHS = 2


def split_available(available: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Everything before the eval window, cut into a fit part and the most recent
    two months, which are used to learn blend weights and calibration."""
    cutoff = available["date"].max() - pd.DateOffset(months=BLEND_MONTHS)
    return available[available.date <= cutoff], available[available.date > cutoff]


def baseline_predictions(fit: pd.DataFrame, target: pd.DataFrame) -> dict:
    """Two naive rate-per-mile rules the model has to beat to justify itself."""
    rpm = fit[D.TARGET] / fit["distance"]
    global_rpm = float(rpm.median())
    lane = rpm.groupby([fit.pickup, fit.delivery]).median()
    lane_rpm = pd.MultiIndex.from_frame(target[["pickup", "delivery"]]).map(lane)
    return {
        "baseline_global_rpm": np.full(len(target), global_rpm) * target["distance"].values,
        "baseline_lane_rpm": np.where(pd.isna(lane_rpm), global_rpm, lane_rpm) * target["distance"].values,
    }


def run_folds(frames: dict, seed: int = 0) -> dict:
    train = frames["train"]
    results = []
    for start, end in FOLDS:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        available = train[train.date < start_ts]
        evaluation = train[(train.date >= start_ts) & (train.date <= end_ts)]
        fit_frame, blend_frame = split_available(available)

        stack = P.fit_stack(fit_frame, blend_frame, seed=seed)
        predicted = stack.predict(evaluation)

        fold = {
            "eval_window": [start, end],
            "fit_rows": int(len(fit_frame)),
            "blend_rows": int(len(blend_frame)),
            "eval_rows": int(len(evaluation)),
            "weights": stack.weights.round(4).to_dict(),
            "calibration": stack.calibration,
            "scores": {"stack": M.metrics(evaluation[D.TARGET], predicted)},
        }
        for name, values in baseline_predictions(available, evaluation).items():
            fold["scores"][name] = M.metrics(evaluation[D.TARGET], values)
        for name, values in stack.base_predictions(evaluation).items():
            fold["scores"][name] = M.metrics(
                evaluation[D.TARGET], M.to_dollars(values, evaluation["distance"], stack.calibration)
            )
        fold["breakdown"] = breakdown(evaluation, predicted)
        results.append(fold)
        print(f"  fold {start}..{end}: stack MAE ${fold['scores']['stack']['mae']:.2f} "
              f"| lane baseline ${fold['scores']['baseline_lane_rpm']['mae']:.2f}")
    return {"folds": results, "summary": summarise(results)}


def breakdown(evaluation: pd.DataFrame, predicted) -> dict:
    """Where the error actually sits - the part a single headline number hides."""
    frame = evaluation.assign(predicted=predicted)
    out = {}
    for label, keys in {"equipment": "equipment", "distance_band": "distance_band"}.items():
        out[label] = {
            str(key): M.metrics(group[D.TARGET], group["predicted"])
            for key, group in frame.groupby(keys, observed=True)
        }
    return out


def summarise(results: list) -> dict:
    names = results[0]["scores"].keys()
    return {
        name: {
            metric: float(np.mean([fold["scores"][name][metric] for fold in results]))
            for metric in ("mae", "rmse", "mape", "medape", "r2")
        }
        for name in names
    }


def random_split_reference(frames: dict, seed: int = 0) -> dict:
    """The optimism check: same model, same features, rows shuffled instead of
    split by time. Reported only as a contrast."""
    train = frames["train"].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    cut = int(len(train) * 0.8)
    fit_frame, rest = train.iloc[:cut], train.iloc[cut:]
    blend_frame, evaluation = rest.iloc[: len(rest) // 2], rest.iloc[len(rest) // 2:]
    stack = P.fit_stack(fit_frame, blend_frame, seed=seed, refit_on_all=False)
    return M.metrics(evaluation[D.TARGET], stack.predict(evaluation))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling-origin validation of the freight rate stack.")
    parser.add_argument("--sample", type=int, default=0, help="row subsample for a fast smoke run")
    parser.add_argument("--skip-random-split", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frames = P.prepare()
    if args.sample:
        frames["train"] = frames["train"].sample(args.sample, random_state=args.seed).sort_values("date")

    print("Rolling-origin folds (two-month future blocks):")
    report = run_folds(frames, seed=args.seed)
    if not args.skip_random_split:
        print("Random-split reference (for the optimism gap):")
        report["random_split_reference"] = random_split_reference(frames, seed=args.seed)
        print(f"  random split MAE ${report['random_split_reference']['mae']:.2f}")
    report["data_quality"] = D.quality_report(frames["raw_train"], frames["raw_validation"])

    D.REPORTS.mkdir(parents=True, exist_ok=True)
    path = D.REPORTS / "validation_metrics.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {path.relative_to(D.ROOT)}")
    summary = report["summary"]
    print("\nMean across folds (MAE $ / MAPE %):")
    for name in sorted(summary, key=lambda n: summary[n]["mae"]):
        print(f"  {name:22s} {summary[name]['mae']:8.2f}  {summary[name]['mape']:6.2f}")


if __name__ == "__main__":
    main()
