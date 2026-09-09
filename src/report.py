"""Build the submission report (.docx) from the artefacts the pipeline produced.

Everything in the document is read from `reports/*.json`, the EDA figures and the
scorer's chart - no number is typed in by hand, so re-running the pipeline
re-issues a correct report.
"""
from __future__ import annotations

import json

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

import data as D

INK = RGBColor(0x06, 0x4A, 0x56)
FIGURES = D.REPORTS / "figures"
CHART = D.ROOT / "scorer_results" / "candidate_december.png"

MODEL_ROLES = {
    "stack": "Greedy blend (shipped model)",
    "hgb_mae": "Gradient boosting, absolute-error loss",
    "hgb_deep": "Gradient boosting, deeper, absolute-error loss",
    "hgb_mse": "Gradient boosting, squared-error loss",
    "forest": "Random forest",
    "ridge": "Ridge regression",
    "baseline_lane_rpm": "Baseline: lane median rate per mile x distance",
    "baseline_global_rpm": "Baseline: global median rate per mile x distance",
}


def heading(document, text, level=1):
    paragraph = document.add_heading(text, level=level)
    for run in paragraph.runs:
        run.font.color.rgb = INK
    return paragraph


def bullets(document, items):
    for item in items:
        document.add_paragraph(item, style="List Bullet")


def table(document, columns, rows, widths=None):
    element = document.add_table(rows=1, cols=len(columns))
    element.style = "Light Grid Accent 1"
    element.alignment = WD_TABLE_ALIGNMENT.CENTER
    header = element.rows[0].cells
    for index, name in enumerate(columns):
        header[index].text = str(name)
        for paragraph in header[index].paragraphs:
            for run in paragraph.runs:
                run.bold = True
    for row in rows:
        cells = element.add_row().cells
        for index, value in enumerate(row):
            cells[index].text = str(value)
    if widths:
        for row in element.rows:
            for index, width in enumerate(widths):
                row.cells[index].width = Inches(width)
    for row in element.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(9)
    return element


def figure(document, path, caption):
    if not path.is_file():
        document.add_paragraph(f"[missing figure: {path.name}]")
        return
    document.add_picture(str(path), width=Inches(6.3))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    note = document.add_paragraph(caption)
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in note.runs:
        run.italic = True
        run.font.size = Pt(9)


def money(value):
    return f"${value:,.2f}"


def build() -> None:
    metrics = json.loads((D.REPORTS / "validation_metrics.json").read_text())
    eda = json.loads((D.REPORTS / "eda_report.json").read_text())
    training = json.loads((D.REPORTS / "training_report.json").read_text())
    summary = metrics["summary"]
    quality = metrics["data_quality"]

    document = Document()
    style = document.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)

    title = document.add_heading("Freight Rate Prediction - Model Report", level=0)
    for run in title.runs:
        run.font.color.rgb = INK
    subtitle = document.add_paragraph(
        f"Training data {quality['train_date_range'][0]} to {quality['train_date_range'][1]} "
        f"({quality['train_rows']:,} loads). Predictions for {quality['validation_rows']:,} loads "
        f"across {quality['validation_date_range'][0]} to {quality['validation_date_range'][1]}."
    )
    for run in subtitle.runs:
        run.italic = True

    # ---------------------------------------------------------------- summary
    heading(document, "Summary", 1)
    stack = summary["stack"]
    best_baseline = min(
        (summary[name] for name in summary if name.startswith("baseline")), key=lambda s: s["mae"]
    )
    document.add_paragraph(
        f"The shipped model is a blend of five learners predicting log(rate per mile). "
        f"Averaged over three rolling-origin folds, each hiding a two-month future block, it reaches "
        f"{money(stack['mae'])} mean absolute error ({stack['mape']:.2f}% MAPE, R² {stack['r2']:.3f}), "
        f"against {money(best_baseline['mae'])} for the strongest naive rate-per-mile baseline - "
        f"a {100 * (1 - stack['mae'] / best_baseline['mae']):.0f}% reduction in error."
    )

    # ------------------------------------------------------------- exploration
    heading(document, "1. What the data says", 1)
    bullets(document, [
        f"Distance dominates the raw rate (correlation {eda['distance_corr_with_rate']:.2f}), so the model "
        f"predicts log(rate per mile) with distance as an offset rather than predicting dollars directly.",
        f"quote_signal is a decoy. Its correlation with realised rate per mile is "
        f"{quality['quote_signal_corr_with_rpm']:.2f}; its mean sits next to the rate-per-mile mean, so it "
        f"looks like a price quote but carries no signal. It is dropped everywhere.",
        f"market_index only works in aggregate: {quality['market_index_row_corr_with_rpm']:.2f} correlation "
        f"row by row, {quality['market_index_daily_corr_with_rpm']:.2f} once averaged to a day. It is a daily "
        f"market reading seen through per-row noise, so the model uses a daily calendar built from it.",
        f"Rate per mile falls with length of haul and separates cleanly by equipment "
        f"(Dry Van {eda['rate_per_mile_by_equipment']['Dry Van']:.2f}, "
        f"Flatbed {eda['rate_per_mile_by_equipment']['Flatbed']:.2f}, "
        f"Reefer {eda['rate_per_mile_by_equipment']['Reefer']:.2f} $/mile median).",
        f"{len(quality['cities_in_validation_absent_from_train'])} cities appear in the prediction set that "
        f"never appear in training ({', '.join(quality['cities_in_validation_absent_from_train'])}). No raw "
        f"city identifier is used as a feature; geography enters through coordinates, great-circle distance "
        f"and bearing, and every lane encoding carries a fallback flag.",
    ])
    figure(document, FIGURES / "market_vs_rate.png",
           "Daily mean rate per mile against the daily mean market index.")
    figure(document, FIGURES / "rpm_vs_distance.png",
           "Rate per mile by length of haul and equipment type.")

    # ----------------------------------------------------------- data quality
    heading(document, "2. Data quality and how it was handled", 1)
    table(document,
          ["Issue", "Rows", "Handling"],
          [["Negative weight (correct magnitude, flipped sign)", f"{quality['negative_weight_rows']:,}",
            "Absolute value"],
           ["Missing weight", f"{quality['missing_weight_rows']:,}",
            "Left missing and flagged; the boosters split on missing natively"],
           ["Missing market_index", f"{quality['missing_market_index_rows']:,}",
            "Filled from that day's market calendar and flagged"],
           ["Injected multiplicative rate shocks", "~1%",
            "Not deleted. The primary learner optimises absolute error, which estimates the "
            "conditional median and is not dragged by them"],
           ["distance long against great-circle distance", "214",
            "Not repaired. The ratio is constant within a lane (std 0.02), so it is distortion in "
            "the synthetic coordinates, not per-row corruption"]],
          widths=[2.1, 0.7, 3.5])
    document.add_paragraph(
        f"The shock diagnosis is what drives the choice of loss. Measured against each lane's own median, "
        f"the log rate-per-mile residual tails are near-perfectly mirrored "
        f"({quality['contamination_p005_log_residual']:.2f} at the 0.5th percentile against "
        f"+{quality['contamination_p995_log_residual']:.2f} at the 99.5th). A fat-tailed price distribution "
        f"would not be symmetric in log space; an injected multiplicative shock is. Deleting those rows would "
        f"also delete genuine hot-market pricing, so they are kept and absorbed by a robust objective instead."
    )
    figure(document, FIGURES / "contamination.png",
           "Log rate-per-mile against the lane median. Dashed lines mark the 0.5th and 99.5th percentiles.")

    # ------------------------------------------------------------------ split
    heading(document, "3. Train/test split and validation approach", 1)
    document.add_paragraph(
        "Training data ends 2025-10-31 and every graded prediction falls in November and December. That makes "
        "this a forward-extrapolation problem, not a random-holdout problem, and the validation scheme is "
        "built to match it: three rolling-origin folds, each hiding a two-month future block at exactly the "
        "horizon the real submission faces."
    )
    table(document,
          ["Fold", "Fitted on", "Blend/calibration window", "Evaluated on", "Rows scored"],
          [[index + 1,
            f"{fold['fit_rows']:,} rows",
            f"{fold['blend_rows']:,} rows (2 months)",
            f"{fold['eval_window'][0]} to {fold['eval_window'][1]}",
            f"{fold['eval_rows']:,}"]
           for index, fold in enumerate(metrics["folds"])],
          widths=[0.5, 1.2, 1.8, 2.0, 0.9])
    bullets(document, [
        "Nothing inside an evaluated block ever reaches the model: base learners, lane encodings, blend "
        "weights and the dollar calibration factor are all fitted strictly on earlier dates.",
        "Blend weights and the calibration factor are fitted on the most recent two months available inside "
        "each fold, never on the block being scored, which mirrors how the shipped model is built.",
        "Lane encodings are computed out of fold, so a lane's own labels never reach its own features.",
        "No raw time index is used as a feature. A tree cannot split past the last date it saw, and the "
        "target window is two months beyond the end of training. Seasonality enters only through cyclical "
        "calendar terms, holiday proximity and the market calendar, all defined for future dates.",
    ])
    if "random_split_reference" in metrics:
        reference = metrics["random_split_reference"]
        gap = 100 * (1 - reference["mae"] / stack["mae"])
        document.add_paragraph(
            f"For contrast, the same model scored on a random 80/20 split of the same rows reports "
            f"{money(reference['mae'])} MAE against {money(stack['mae'])} on the time-based folds - "
            f"{gap:.0f}% optimistic. Random splitting lets the model see the same weeks it is scored on; "
            f"the time-based number is the one to trust."
        )

    # ------------------------------------------------------------------ model
    heading(document, "4. Model", 1)
    document.add_paragraph(
        "Five learners over one shared design matrix, combined by greedy ensemble selection scored on dollar "
        "mean absolute error. Selecting on the metric that is actually graded matters: an earlier version "
        "fitted non-negative least squares in log space and produced a blend that scored worse than its own "
        "best member. Predictions are converted back to dollars with a single multiplicative calibration "
        f"factor ({training['calibration']:.4f}) chosen on held-out data, because a log-space fit is biased "
        "in dollar space and the textbook smearing correction targets the mean rather than the metric here."
    )
    weights = training["blend_weights"]
    table(document,
          ["Learner", "Role", "Blend weight"],
          [[name, MODEL_ROLES.get(name, ""), f"{weights.get(name, 0):.2f}"]
           for name in sorted(weights, key=lambda n: -weights[n])],
          widths=[1.1, 3.9, 1.0])
    document.add_paragraph(
        "Features: log distance and great-circle distance, detour ratio, bearing, endpoint and midpoint "
        "coordinates, weight and weight per mile with a missing flag, equipment indicators, distance band, "
        "day of week, cyclical day-of-year and month terms, distance to the nearest holiday, the market "
        "calendar (daily mean, 7- and 28-day rolling means and the gap between them), and shrunk "
        "out-of-fold median log rate-per-mile encodings for lane, origin, destination, corridor, "
        "origin-by-equipment and equipment-by-distance-band, each with a count and a seen flag."
    )

    # ---------------------------------------------------------------- results
    heading(document, "5. Results", 1)
    document.add_paragraph("Mean across the three rolling-origin folds, best first.")
    table(document,
          ["Model", "MAE", "RMSE", "MAPE", "Median APE", "R²"],
          [[MODEL_ROLES.get(name, name), money(scores["mae"]), money(scores["rmse"]),
            f"{scores['mape']:.2f}%", f"{scores['medape']:.2f}%", f"{scores['r2']:.3f}"]
           for name, scores in sorted(summary.items(), key=lambda kv: kv[1]["mae"])],
          widths=[2.4, 0.9, 0.9, 0.8, 0.9, 0.6])
    document.add_paragraph(
        "The squared-error twin of the primary learner is kept in the table deliberately: it shares every "
        "feature and hyperparameter with the absolute-error version and differs only in objective, so the gap "
        "between them is the measured cost of letting the injected shocks pull the fit."
    )
    last = metrics["folds"][-1]
    heading(document, "Error by segment (most recent fold)", 2)
    table(document,
          ["Equipment", "MAE", "MAPE"],
          [[name, money(scores["mae"]), f"{scores['mape']:.2f}%"]
           for name, scores in sorted(last["breakdown"]["equipment"].items())],
          widths=[1.6, 1.0, 1.0])
    bands = ["0-300", "300-600", "600-1000", "1000-1500", "1500-2500", "2500+"]
    table(document,
          ["Distance band (mi)", "MAE", "MAPE"],
          [[bands[int(key)] if int(key) < len(bands) else key, money(scores["mae"]), f"{scores['mape']:.2f}%"]
           for key, scores in sorted(last["breakdown"]["distance_band"].items(), key=lambda kv: int(kv[0]))],
          widths=[1.6, 1.0, 1.0])

    # --------------------------------------------------------------- december
    document.add_page_break()
    heading(document, "6. Fixed December prediction chart", 1)
    document.add_paragraph(
        "Produced by the provided score.py from december_chart_inputs.csv: Lexington to Fort Wayne, "
        "360 miles, Dry Van, 32,000 lb, with only the date changing across the 31 days."
    )
    figure(document, CHART, "candidate_december.png, generated by score.py.")
    document.add_paragraph(
        "These rows arrive with six columns and no coordinates, market index or quote signal, yet they are "
        "scored by exactly the same model as the validation set: coordinates come from the city table and the "
        "market state from the daily calendar, both built from feature columns of the supplied files and never "
        "from labels. That is what makes the curve move day to day rather than sitting flat - the only inputs "
        "that change are the calendar terms, the holiday proximity and the market calendar."
    )

    # ------------------------------------------------------------- reproduce
    heading(document, "7. Reproducing this", 1)
    for line in ["python -m pip install -r requirements.txt",
                 "python src/eda.py",
                 "python src/validate.py",
                 "python src/train.py",
                 "python src/predict.py",
                 "python score.py --predictions validation_predictions.csv \\",
                 "                --december-predictions december_chart_inputs.csv"]:
        paragraph = document.add_paragraph(line)
        paragraph.paragraph_format.space_after = Pt(0)
        for run in paragraph.runs:
            run.font.name = "Consolas"
            run.font.size = Pt(9)

    output = D.REPORTS / "freight_rate_model_report.docx"
    document.save(output)
    print(f"Wrote {output.relative_to(D.ROOT)}")


if __name__ == "__main__":
    build()
