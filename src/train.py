"""Fit the shipped model.

Base learners are fitted on January-August, the blend weights and the dollar
calibration are learned on the held-out September-October block, and the base
learners are then refitted on all ten months so the shipped model has seen the
weeks closest to the November-December prediction window.
"""
from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd

import data as D
import pipeline as P

BLEND_START = "2025-09-01"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the freight rate stack.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frames = P.prepare()
    train = frames["train"]
    cut = pd.Timestamp(BLEND_START)
    fit_frame = train[train.date < cut]
    blend_frame = train[train.date >= cut]
    print(f"fit rows {len(fit_frame):,} (through {cut.date() - pd.Timedelta(days=1)}) | "
          f"blend rows {len(blend_frame):,} (from {cut.date()})")

    stack = P.fit_stack(fit_frame, blend_frame, seed=args.seed)

    D.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(stack, D.ARTIFACTS / "stack.joblib")
    report = {
        "blend_weights": stack.weights.round(4).to_dict(),
        "calibration": stack.calibration,
        "holdout_window": [BLEND_START, str(train.date.max().date())],
        "holdout_scores": stack.holdout_report,
    }
    (D.REPORTS / "training_report.json").write_text(json.dumps(report, indent=2))

    print("\nHeld-out September-October scores (MAE $):")
    for name, scores in sorted(stack.holdout_report.items(), key=lambda kv: kv[1]["mae"]):
        print(f"  {name:10s} {scores['mae']:8.2f}   MAPE {scores['mape']:5.2f}%")
    print("\nblend weights:", stack.weights.round(3).to_dict())
    print("calibration:", round(stack.calibration, 4))
    print(f"saved {(D.ARTIFACTS / 'stack.joblib').relative_to(D.ROOT)}")


if __name__ == "__main__":
    main()
