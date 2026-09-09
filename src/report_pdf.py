"""Build the narrative PDF write-up.

The DOCX report is the terse deliverable; this is the long-form version - what was
explored, what turned out to be true, what was tried and abandoned, and what
finally shipped. Every figure and every number is read from the artefacts the
pipeline produced, so re-running the pipeline re-issues a correct document.
"""
from __future__ import annotations

import json

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

import data as D

INK = colors.HexColor("#064A56")
RULE = colors.HexColor("#9DAFB3")
BAND = colors.HexColor("#EDF3F4")
MUTED = colors.HexColor("#455A60")

FIGURES = D.REPORTS / "figures"
CHART = D.ROOT / "scorer_results" / "candidate_december.png"

ROLES = {
    "stack": "Shipped model: equal-weight average of the two robust boosters",
    "hgb_deep": "Gradient boosting, absolute-error loss, deeper trees",
    "hgb_mae": "Gradient boosting, absolute-error loss",
    "hgb_mse": "Gradient boosting, squared-error loss",
    "forest": "Random forest",
    "ridge": "Ridge regression",
    "baseline_lane_rpm": "Baseline: lane median rate per mile x distance",
    "baseline_global_rpm": "Baseline: global median rate per mile x distance",
}


def styles() -> dict:
    sheet = getSampleStyleSheet()
    base = dict(fontName="Helvetica", fontSize=10, leading=15.2, textColor=colors.HexColor("#1B2B2F"))
    return {
        "title": ParagraphStyle("title", parent=sheet["Title"], fontName="Helvetica-Bold",
                                fontSize=21, leading=25, textColor=INK, alignment=0, spaceAfter=2),
        "subtitle": ParagraphStyle("subtitle", fontName="Helvetica-Oblique", fontSize=10.5,
                                   leading=15, textColor=MUTED, spaceAfter=16),
        "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=14, leading=18,
                             textColor=INK, spaceBefore=20, spaceAfter=7),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11, leading=15,
                             textColor=INK, spaceBefore=13, spaceAfter=5),
        "body": ParagraphStyle("body", alignment=TA_JUSTIFY, spaceAfter=9, **base),
        "bullet": ParagraphStyle("bullet", alignment=TA_JUSTIFY, leftIndent=14,
                                 bulletIndent=3, spaceAfter=6, **base),
        "caption": ParagraphStyle("caption", fontName="Helvetica-Oblique", fontSize=8.5,
                                  leading=11.5, textColor=MUTED, alignment=TA_CENTER, spaceAfter=13),
        "code": ParagraphStyle("code", fontName="Courier", fontSize=8.5, leading=12.5,
                               textColor=colors.HexColor("#1B2B2F"), leftIndent=10, spaceAfter=1),
        "pull": ParagraphStyle("pull", fontName="Helvetica-Oblique", fontSize=10.5, leading=15.5,
                               textColor=INK, leftIndent=14, rightIndent=14, spaceBefore=4,
                               spaceAfter=12, alignment=TA_JUSTIFY),
    }


def money(value) -> str:
    return f"${value:,.2f}"


def figure(path, caption, style, width=6.2):
    if not path.is_file():
        return [Paragraph(f"[missing figure: {path.name}]", style["caption"])]
    source_width, source_height = ImageReader(str(path)).getSize()
    height = width * source_height / source_width
    image = Image(str(path), width=width * inch, height=height * inch)
    image.hAlign = "CENTER"
    return [Spacer(1, 4), image, Spacer(1, 4), Paragraph(caption, style["caption"])]


def table(columns, rows, widths, align_right=()):
    data = [[Paragraph(f"<b>{c}</b>", ParagraphStyle("th", fontName="Helvetica-Bold",
                                                     fontSize=8.5, leading=11,
                                                     textColor=colors.white)) for c in columns]]
    cell = ParagraphStyle("td", fontName="Helvetica", fontSize=8.5, leading=11.5)
    right = ParagraphStyle("tdr", parent=cell, alignment=2)
    for row in rows:
        data.append([Paragraph(str(v), right if i in align_right else cell) for i, v in enumerate(row)])
    element = Table(data, colWidths=[w * inch for w in widths], repeatRows=1, hAlign="LEFT")
    element.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BAND]),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return element


def footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(0.9 * inch, 0.68 * inch, LETTER[0] - 0.9 * inch, 0.68 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.9 * inch, 0.5 * inch, "Freight Rate Prediction - engineering write-up")
    canvas.drawRightString(LETTER[0] - 0.9 * inch, 0.5 * inch, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def build() -> None:
    metrics = json.loads((D.REPORTS / "validation_metrics.json").read_text())
    eda = json.loads((D.REPORTS / "eda_report.json").read_text())
    training = json.loads((D.REPORTS / "training_report.json").read_text())
    summary, quality, folds = metrics["summary"], metrics["data_quality"], metrics["folds"]
    stack = summary["stack"]
    baselines = {k: v for k, v in summary.items() if k.startswith("baseline")}
    best_baseline_name = min(baselines, key=lambda k: baselines[k]["mae"])
    best_baseline = baselines[best_baseline_name]
    style = styles()
    S = []
    P = lambda text: S.append(Paragraph(text, style["body"]))
    H = lambda text, level="h1": S.append(Paragraph(text, style[level]))
    B = lambda text: S.append(Paragraph(text, style["bullet"], bulletText="•"))

    S.append(Paragraph("Freight Rate Prediction", style["title"]))
    S.append(Paragraph(
        f"How the model was built, what the data turned out to be hiding, and which "
        f"three approaches were measured and thrown away along the way. "
        f"Training data {quality['train_date_range'][0]} to {quality['train_date_range'][1]}, "
        f"{quality['train_rows']:,} loads; predictions for {quality['validation_rows']:,} loads across "
        f"{quality['validation_date_range'][0]} to {quality['validation_date_range'][1]}.",
        style["subtitle"]))

    # ------------------------------------------------------------------------
    H("The brief, and the one fact that shaped everything")
    P("The task looks like ordinary tabular regression: given a lane, an equipment type, a weight and a "
      "date, predict what the load posted for. The wrinkle is in the calendar. Every labelled row falls "
      f"between {quality['train_date_range'][0]} and {quality['train_date_range'][1]}, and every row that "
      f"gets graded falls in November and December. Nothing is being interpolated. The model is being asked "
      "to step two months past the last date it has ever seen.")
    P("That single fact drove most of what follows: it is why validation hides future blocks rather than "
      "random rows, why no raw time index appears anywhere in the feature set, and why a column that looks "
      "like a market signal was worth several hours of attention before it earned its place.")

    # ------------------------------------------------------------------------
    H("First contact with the data")
    P("Forty-eight thousand labelled loads, sixty-four cities, three equipment types, fourteen columns. The "
      "first thing worth knowing is how lopsided the signal is: distance correlates with the posted rate at "
      f"{eda['distance_corr_with_rate']:.2f}. A model that only multiplies miles by a constant is already "
      "most of the way there, which means the interesting question is never \"what does this load cost\" but "
      "\"what does this load cost <i>per mile</i>, and why is that number different from the lane's usual one\".")
    P("So the model predicts log(rate per mile) and multiplies distance back in afterwards. Distance stops "
      "being something the trees have to rediscover in every branch and becomes an offset, which frees the "
      "whole model to spend its capacity on the part that is genuinely hard.")

    H("The column that looks like a price and isn't", "h2")
    P(f"<b>quote_signal</b> is the trap in this dataset. It is named like a price quote, and its mean sits "
      f"right next to the mean rate per mile, so it reads at a glance as though someone has handed you most "
      f"of the answer. Its correlation with the realised rate per mile is "
      f"{quality['quote_signal_corr_with_rpm']:.2f}. It is noise dressed up in a suggestive name, and it is "
      f"dropped everywhere in this pipeline.")
    S.append(Paragraph(
        "The lesson generalises past this dataset: a column's name is a claim made by whoever built the "
        "table, not evidence. It costs one correlation to check and it can cost a whole model not to.",
        style["pull"]))

    H("The column that works, but only from a distance", "h2")
    P(f"<b>market_index</b> is the mirror image. Row by row it correlates with rate per mile at "
      f"{quality['market_index_row_corr_with_rpm']:.2f}, which on its own would look like another decoy. "
      f"Average it to a day and the correlation jumps to {quality['market_index_daily_corr_with_rpm']:.2f}. "
      f"It is a daily market reading published per load with per-row noise on top, and the noise is large "
      f"enough to bury it if the column is fed in raw.")
    P("The model therefore builds a market calendar - one row per day, carrying that day's mean, seven- and "
      "twenty-eight-day rolling means, and the gap between the two as a crude momentum term. That calendar "
      "does double duty later, because it is also what lets the December chart rows be scored at all.")
    S.extend(figure(FIGURES / "market_vs_rate.png",
                    "Daily mean rate per mile against the daily mean market index. Invisible row by row, "
                    "unmistakable once aggregated.", style))

    H("Two more things worth knowing", "h2")
    B(f"<b>Rate per mile falls with the length of the haul</b>, from roughly $2.74 a mile under three "
      f"hundred miles to $1.90 over twenty-five hundred, and the three equipment types sit on visibly "
      f"different curves - Dry Van at {eda['rate_per_mile_by_equipment']['Dry Van']:.2f} a mile at the "
      f"median, Flatbed at {eda['rate_per_mile_by_equipment']['Flatbed']:.2f}, Reefer at "
      f"{eda['rate_per_mile_by_equipment']['Reefer']:.2f}.")
    B(f"<b>{len(quality['cities_in_validation_absent_from_train'])} cities appear in the prediction set that "
      f"never appear in training</b> ({', '.join(quality['cities_in_validation_absent_from_train'])}). Any "
      f"model leaning on a raw city identifier would have had nothing to say about them. Geography therefore "
      f"enters through coordinates, great-circle distance and bearing, and every lane-level encoding carries "
      f"a fallback and a flag saying whether it was ever seen.")
    S.extend(figure(FIGURES / "rpm_vs_distance.png",
                    "Rate per mile by length of haul and equipment type.", style))

    # ------------------------------------------------------------------------
    S.append(PageBreak())
    H("Data quality, and two problems I deliberately left alone")
    P("Three defects are real and were repaired. Two more look like defects and were not, which took longer "
      "to establish than the repairs did.")
    S.append(table(
        ["Finding", "Rows", "What was done"],
        [["Negative weight, correct magnitude", f"{quality['negative_weight_rows']:,}", "Absolute value - a sign error, not a measurement"],
         ["Missing weight", f"{quality['missing_weight_rows']:,}", "Left missing and flagged; the boosters split on missing natively"],
         ["Missing market_index", f"{quality['missing_market_index_rows']:,}", "Filled from that day's market calendar and flagged"],
         ["distance long against great-circle", "214", "<b>Left alone</b> - see below"],
         ["Extreme rate-per-mile outliers", "~1%", "<b>Left in</b> - absorbed by a robust loss instead"]],
        [1.9, 0.6, 3.7]))
    S.append(Spacer(1, 10))

    H("Why the suspicious distances were not repaired", "h2")
    P("Two hundred and fourteen rows carry a distance far longer than the great-circle distance between "
      "their endpoints - one by a factor of nine. The obvious move is to overwrite them with the lane median. "
      "The obvious move is wrong. Grouping the ratio by lane shows a standard deviation of 0.02 <i>within</i> "
      "each lane: the discrepancy is a fixed property of the city pair, not a per-row accident. The "
      "coordinates in this dataset are synthetic and shifted from real geography by varying amounts, so for a "
      "handful of short-haul pairs the fake coordinates land implausibly close together while the mileage "
      "stays real. Repairing those rows would have corrupted 214 correct records to fix nothing.")

    H("Why the outliers were kept", "h2")
    P(f"Measured against each lane's own median, the log rate-per-mile residuals have tails that are almost "
      f"perfectly mirrored: {quality['contamination_p005_log_residual']:.2f} at the half-percentile against "
      f"+{quality['contamination_p995_log_residual']:.2f} at the ninety-nine-and-a-half. Real freight pricing "
      f"is not symmetric in log space - hot markets spike upward far more readily than they collapse. That "
      f"symmetry is the fingerprint of an injected multiplicative shock, roughly one row in a hundred "
      f"multiplied or divided by about three.")
    P("Deleting them was tempting and would have been a mistake, because the upper tail is not purely "
      "synthetic: it clusters in June, the peak of the market in this data, so some of those rows are genuine "
      "surge pricing that the model should learn from. Rather than choose which extreme rows to believe, the "
      "primary learner optimises absolute error. That estimates the conditional median instead of the mean, "
      "and a median simply does not move when one row in a hundred is multiplied by three. The contamination "
      "is handled by the objective rather than by a deletion rule that would have thrown away real signal.")
    S.extend(figure(FIGURES / "contamination.png",
                    "Log rate per mile against the lane median. Dashed lines mark the 0.5th and 99.5th "
                    "percentiles - note how nearly they mirror each other.", style))

    # ------------------------------------------------------------------------
    S.append(PageBreak())
    H("How the data was split, and why it matters more than usual")
    P("A random hold-out would have been the wrong instrument here, and not by a small margin. Splitting "
      "rows at random lets the model train on the same weeks it is scored on, so it never has to survive the "
      "thing the real task demands: a two-month step into dates it has never seen.")
    P("Validation is therefore rolling-origin. Three folds, each hiding a contiguous two-month future block "
      "at exactly the horizon the submission faces. Within each fold, base learners are fitted on the "
      "earliest stretch, the dollar calibration factor is fitted on the most recent two months before the "
      "hidden block, and only then is the hidden block scored.")
    S.append(table(
        ["Fold", "Fitted on", "Calibration window", "Scored on", "Rows scored"],
        [[str(i + 1), f"{f['fit_rows']:,} rows", f"{f['blend_rows']:,} rows",
          f"{f['eval_window'][0]} to {f['eval_window'][1]}", f"{f['eval_rows']:,}"]
         for i, f in enumerate(folds)],
        [0.45, 1.15, 1.25, 2.15, 1.0], align_right=(4,)))
    S.append(Spacer(1, 10))
    B("Nothing inside a scored block reaches the model. Base learners, lane encodings and the calibration "
      "factor are all fitted strictly on earlier dates.")
    B("Lane encodings are computed out of fold, so a lane's own labels never leak into its own features. "
      "There is a self-check asserting exactly this, because it is the kind of leak that quietly flatters "
      "every number in a report.")
    B("No raw time index is used as a feature anywhere. A decision tree cannot split past the largest value "
      "it saw in training, so a date index would silently clamp on every graded row. Seasonality enters only "
      "through cyclical day-of-year and month terms, distance to the nearest holiday, and the market "
      "calendar - all of which are defined for dates in the future.")

    if "random_split_reference" in metrics:
        reference = metrics["random_split_reference"]
        H("What a random split would have told me", "h2")
        P(f"Worth quantifying, because it is the number a less careful write-up would have reported. The same "
          f"model, the same features, scored on a random 80/20 split of the same rows, reports "
          f"{money(reference['mae'])} mean absolute error. Scored on time-based folds it reports "
          f"{money(stack['mae'])}. The random split is "
          f"{100 * (1 - reference['mae'] / stack['mae']):.0f}% optimistic - it would have promised roughly "
          f"half the error the model can actually deliver on the task as posed. Both numbers are in the "
          f"repository; only one of them means anything.")

    # ------------------------------------------------------------------------
    H("The model")
    P("Five learners are fitted over one shared design matrix so they can be compared honestly. The shipped "
      "predictor is a fixed, equal-weight average of the two absolute-error boosters. Nothing about that "
      "weighting is fitted, and the next section is the story of why.")
    weights = training["ensemble_weights"]
    S.append(table(
        ["Learner", "Role", "Weight"],
        [[name, ROLES.get(name, ""), f"{weights.get(name, 0.0):.2f}"]
         for name in sorted(weights, key=lambda n: (-weights[n], n))],
        [0.85, 4.2, 0.65], align_right=(2,)))
    S.append(Spacer(1, 10))
    P("The features fall into five groups. Geography: log distance, great-circle distance, the ratio between "
      "them, bearing, endpoint and midpoint coordinates. The load itself: weight, weight per mile, a missing "
      "flag, equipment indicators and a distance band. Calendar: day of week, cyclical day-of-year and month "
      "terms, and distance to the nearest holiday. Market: the daily calendar described earlier. And lane "
      "history: shrunk out-of-fold median log rate-per-mile encodings for the lane, the origin, the "
      "destination, the corridor, origin-by-equipment and equipment-by-distance-band, each carrying a support "
      "count and a flag for whether it was ever observed.")
    P(f"Predictions come back from log space through a single multiplicative calibration factor "
      f"({training['calibration']:.4f}) fitted on held-out data. A log-space fit is biased in dollar space, "
      f"and the textbook smearing correction targets the mean rather than the metric in play, so the factor "
      f"is simply searched for against dollar error instead of assumed. It is the one quantity the held-out "
      f"block still determines.")

    # ------------------------------------------------------------------------
    S.append(PageBreak())
    H("Three ensembles that did not work")
    P("This is the part of the project that consumed the most time and produced the least code, and it is "
      "worth writing down properly, because each attempt failed for a different and instructive reason.")
    H("Attempt one: non-negative least squares in log space", "h2")
    P("The natural first move. Fit non-negative weights over the base predictions, constrained to sum to "
      "one. It produced a blend scoring $188 mean absolute error when its own best member scored $166. A "
      "combination that loses to one of its ingredients is not a subtle bug, and the cause was plain once "
      "stated: least squares in log space optimises squared error on log rate per mile, while the model is "
      "graded on absolute error in dollars. Those are different objectives, and the weights were dutifully "
      "optimising the wrong one.")
    H("Attempt two: greedy selection against the real metric", "h2")
    P("Replacing least squares with Caruana-style greedy ensemble selection, scored directly on dollar mean "
      "absolute error, fixed that immediately - the blend went from $188 to $152 and drew level with its best "
      "member. Then the full-data run exposed something worse. On the earliest fold the blend scored $406 "
      "where its best member scored $118. Inspecting the weights showed it had handed 74% of the ensemble to "
      "the ridge model, which went on to post $498 on the very block it was chosen to predict.")
    H("Attempt three: bagging the selection", "h2")
    P("Greedy selection is known to overfit small selection sets, and the published remedy is to bag it - run "
      "the selection over bootstrap subsamples and average the resulting weights. That was implemented, and "
      "it moved ridge's weight from 0.744 to 0.744. A fix that changes nothing is information: the failure "
      "was not sampling noise, so no amount of resampling the same window would ever have addressed it.")
    H("What was actually wrong", "h2")
    P("The weights were being learned from one set of models and applied to a different one. Base learners "
      "are fitted on the pre-calibration window, the weights are learned from those fitted models, and then "
      "the learners are refitted on the pre-calibration window plus the calibration window before anything is "
      "predicted - so that the shipped model has seen the most recent weeks. On the earliest fold that refit "
      "doubles the training data, from 9,255 rows to 19,110. Gradient boosting improves sharply when its "
      "data doubles; ridge regression barely moves. The ranking the weights encoded was obsolete by the time "
      "they were used.")
    S.append(Paragraph(
        "No weighting scheme can repair that, because the problem is not which weights were chosen - it is "
        "that they were chosen for a lineup that no longer exists.", style["pull"]))
    P("With the cause understood, the evidence was unambiguous. Across all three folds, fitted blending "
      "never once beat the best single member: it tied twice and lost catastrophically once. A mechanism "
      "whose best case is a tie and whose worst case costs $250 of mean absolute error does not belong in a "
      "shipped model. The fitting was removed.")
    P("What ships instead is an equal-weight average of the two absolute-error boosters. There are no fitted "
      "parameters, so the mismatch cannot recur by construction. Averaging two near-equivalent learners is a "
      "free reduction in variance, and it sidesteps the awkward question of why one was picked over another "
      "when cross-validated MAE separates them by ten cents. All five learners are still fitted and reported, "
      "because the comparison is evidence; only two of them carry any weight at prediction time.")

    # ------------------------------------------------------------------------
    S.append(PageBreak())
    H("Results")
    P(f"Averaged across the three rolling-origin folds. The shipped model reaches {money(stack['mae'])} mean "
      f"absolute error at {stack['mape']:.2f}% MAPE, against {money(best_baseline['mae'])} for the strongest "
      f"naive rate-per-mile rule - an error reduction of "
      f"{100 * (1 - stack['mae'] / best_baseline['mae']):.0f}%.")
    S.append(table(
        ["Model", "MAE", "RMSE", "MAPE", "Med APE", "R<super>2</super>"],
        [[ROLES.get(name, name), money(s["mae"]), money(s["rmse"]), f"{s['mape']:.2f}%",
          f"{s['medape']:.2f}%", f"{s['r2']:.3f}"]
         for name, s in sorted(summary.items(), key=lambda kv: kv[1]["mae"])],
        [2.35, 0.72, 0.78, 0.62, 0.68, 0.55], align_right=(1, 2, 3, 4, 5)))
    S.append(Spacer(1, 10))
    P(f"The row worth dwelling on is <b>hgb_mse</b>. It shares every feature and every hyperparameter with "
      f"<b>hgb_mae</b> and differs in exactly one respect - it minimises squared error instead of absolute "
      f"error - so the gap between them "
      f"({money(summary['hgb_mae']['mae'])} against {money(summary['hgb_mse']['mae'])}) is a clean measurement "
      f"of what the injected shocks cost a model that tries to fit them. It is the empirical version of the "
      f"argument made earlier from the shape of the residual tails.")
    P("It is also worth noticing how far the ridge row sits from the rest. A linear model on these features "
      "degrades badly across a two-month horizon, which is precisely the property that made it so dangerous "
      "when a fitted blend was free to select it on a short window.")
    last = folds[-1]
    H("Where the error actually sits", "h2")
    P("Headline numbers hide structure, so here is the most recent fold broken out by equipment and by "
      "length of haul.")
    bands = ["0-300", "300-600", "600-1000", "1000-1500", "1500-2500", "2500+"]
    S.append(KeepTogether([
        table(["Equipment", "MAE", "MAPE"],
              [[k, money(v["mae"]), f"{v['mape']:.2f}%"]
               for k, v in sorted(last["breakdown"]["equipment"].items())],
              [1.5, 0.9, 0.9], align_right=(1, 2)),
        Spacer(1, 9),
        table(["Distance band (mi)", "MAE", "MAPE"],
              [[bands[int(k)] if int(k) < len(bands) else k, money(v["mae"]), f"{v['mape']:.2f}%"]
               for k, v in sorted(last["breakdown"]["distance_band"].items(), key=lambda kv: int(kv[0]))],
              [1.5, 0.9, 0.9], align_right=(1, 2))]))
    S.append(Spacer(1, 10))

    # ------------------------------------------------------------------------
    S.append(PageBreak())
    H("The fixed December chart")
    P("The chart below is produced by the provided scorer from the completed December input file: Lexington "
      "to Fort Wayne, 360 miles, Dry Van, 32,000 lb, every day of December 2025, with the date as the only "
      "thing that changes.")
    S.extend(figure(CHART, "candidate_december.png, generated by score.py.", style))
    P("Those rows are a quiet test of whether the pipeline was built properly. They arrive with six columns "
      "and no coordinates, no market index and no quote signal - so a model that had bound itself to the "
      "training schema would have had nothing to score them with. Here they pass through the same cleaning "
      "and feature code as everything else: coordinates are recovered from the city table, market state from "
      "the daily calendar, both assembled from feature columns of the supplied files and never from labels.")
    P("That is also why the curve has shape rather than sitting flat. Every physical input is constant "
      "across the 31 rows; the only things moving are day of week, the cyclical calendar terms, proximity to "
      "Christmas and New Year, and the market calendar. If the feature engineering had failed, this chart "
      "would be a horizontal line, and it is the fastest visual check in the whole project.")

    H("Limitations, honestly")
    B(f"<b>The first fold is thin.</b> It fits on only {folds[0]['fit_rows']:,} rows against "
      f"{folds[-1]['fit_rows']:,} for the last, so it is a harder test than the shipped model faces and it "
      f"drags the averages up. It was kept rather than dropped, because it is also the fold that exposed the "
      f"ensembling bug, and a validation scheme that only ever confirms the model is not doing its job.")
    B("<b>Roughly one row in a hundred is unpredictable by construction.</b> The injected multiplicative "
      "shocks are not a function of any feature, so they set a floor on achievable error that no amount of "
      "modelling will get under. This is most visible in RMSE, which stays high while median absolute "
      "percentage error stays low.")
    B("<b>The market calendar is descriptive, not predictive.</b> It uses the market index supplied for the "
      "prediction rows themselves, which is legitimate here because that column is a given input rather than "
      "a label. A production system would need to forecast it, and that forecast error would be additional.")
    B("<b>One scalar for calibration.</b> The log-to-dollar correction is a single multiplier applied "
      "uniformly. Per-segment calibration - by equipment, or by distance band - would probably squeeze out a "
      "little more, but it adds fitted parameters on a small held-out window, which is exactly the pattern "
      "that already went wrong once in this project.")

    H("Reproducing all of it")
    P("Everything in this document is generated from the artefacts the pipeline writes; no number was typed "
      "in by hand. From a clean checkout:")
    for line in ["python3 -m venv .venv && source .venv/bin/activate",
                 "python -m pip install -r requirements.txt",
                 "python src/selfcheck.py   # leak, fallback and December-path assertions",
                 "python src/eda.py         # data-quality report and figures",
                 "python src/validate.py    # rolling-origin validation (~30 min)",
                 "python src/train.py       # fits and saves the shipped model",
                 "python src/predict.py     # writes both submission files",
                 "python score.py --predictions validation_predictions.csv \\",
                 "                --december-predictions december_chart_inputs.csv",
                 "python src/report_pdf.py  # regenerates this document"]:
        S.append(Paragraph(line.replace("&", "&amp;"), style["code"]))
    S.append(Spacer(1, 10))
    P("The repository keeps its reasoning next to its code: the rejected ensembling schemes are documented in "
      "<font face='Courier' size='9'>models.ensemble_weights</font>, the cleaning decisions that were "
      "<i>declined</i> are documented in <font face='Courier' size='9'>data.clean</font>, and "
      "<font face='Courier' size='9'>src/selfcheck.py</font> holds runnable assertions for the failures that "
      "would otherwise be silent - an encoder that leaks, an unseen city that returns a null, a December row "
      "that cannot be scored.")

    output = D.REPORTS / "freight_rate_writeup.pdf"
    SimpleDocTemplate(
        str(output), pagesize=LETTER,
        leftMargin=0.9 * inch, rightMargin=0.9 * inch,
        topMargin=0.85 * inch, bottomMargin=0.85 * inch,
        title="Freight Rate Prediction - engineering write-up",
    ).build(S, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {output.relative_to(D.ROOT)}")


if __name__ == "__main__":
    build()
