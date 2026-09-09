"""Write the two submission files.

The December chart rows arrive with six columns and no coordinates, market index
or quote signal. They are scored by exactly the same model as the validation set:
coordinates come from the city table and the market state from the daily calendar,
both built in `pipeline.prepare`.
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

import data as D
import pipeline as P

DECEMBER_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Write validation and December predictions.")
    parser.add_argument("--model", default=str(D.ARTIFACTS / "stack.joblib"))
    args = parser.parse_args()

    stack = joblib.load(args.model)
    frames = P.prepare()

    validation = frames["validation"]
    predicted = stack.predict(validation)
    predictions = pd.DataFrame({"load_id": validation["load_id"], "predicted_rate": np.round(predicted, 2)})

    # Emit in the template's own row order so the file lines up row for row.
    template = pd.read_csv(D.TEMPLATE_CSV)[["load_id"]]
    predictions = template.merge(predictions, on="load_id", how="left")
    if predictions["predicted_rate"].isna().any():
        raise SystemExit("ERROR: some template load_id values were not predicted")
    predictions.to_csv(D.ROOT / "validation_predictions.csv", index=False)

    december = frames["december"]
    december_out = pd.read_csv(D.DECEMBER_CSV)
    december_out["predicted_rate"] = np.round(stack.predict(december), 2)
    december_out = december_out[DECEMBER_COLUMNS]
    december_out.to_csv(D.ROOT / "december_chart_inputs.csv", index=False)

    print(f"validation_predictions.csv     {len(predictions):,} rows, "
          f"${predictions.predicted_rate.min():,.0f}-${predictions.predicted_rate.max():,.0f}")
    print(f"december_chart_inputs.csv      {len(december_out)} rows, "
          f"${december_out.predicted_rate.min():,.0f}-${december_out.predicted_rate.max():,.0f}")


if __name__ == "__main__":
    main()
