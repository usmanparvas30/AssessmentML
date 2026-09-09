# Freight Rate Prediction

Predicts the posted rate for a truckload shipment. Training data covers
2025-01-01 to 2025-10-31; the graded predictions are the 12,000 November and
December loads in `data/validation.csv`, plus a fixed 31-day December lane used
for the chart.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
python src/selfcheck.py  # leak, fallback and December-path assertions
python src/eda.py        # data-quality report + figures  -> reports/
python src/validate.py   # rolling-origin validation      -> reports/validation_metrics.json
python src/train.py      # fits the shipped model         -> artifacts/stack.joblib
python src/predict.py    # writes both submission files
```

Then the provided scorer:

```bash
python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs.csv
```

Then the two reports (the second needs the scorer's chart to exist):

```bash
python src/report.py      # -> reports/freight_rate_model_report.docx
python src/report_pdf.py  # -> reports/freight_rate_writeup.pdf
```

`src/predict.py` writes the completed December file to the repository root and
leaves the blank original untouched in `data/`, so the run is repeatable.

`src/validate.py` refits the whole stack once per fold and takes roughly 20-30
minutes on a laptop. `--sample 6000` runs the same code path in about two
minutes for a smoke test.

## Approach

**Target.** The model predicts `log(rate / distance)`, not the rate. Distance
alone explains 91% of raw rate variance, so folding it into the target as an
offset leaves the learners free to model the part that is genuinely hard: how
price per mile moves with lane, equipment, market and calendar. Predictions are
converted back with a single multiplicative calibration factor chosen on held-out
data against dollar MAE, because a log-space fit is biased in dollar space and
the textbook smearing correction targets the mean rather than the metric here.

**Validation.** Rolling-origin, three folds, each hiding a two-month *future*
block (May-Jun, Jul-Aug, Sep-Oct) to match the real November-December horizon.
A random 80/20 split is also reported, once, purely to quantify how much it
flatters the model. Blend weights and the calibration factor are fitted only on
the most recent two months available inside each fold, never on the evaluated
block.

**Model.** Five learners are fitted over the same design matrix so they can be
compared honestly, but the shipped model is a fixed, equal-weight average of the
two absolute-error boosters - nothing about the weighting is fitted:

| learner | role |
|---|---|
| `hgb_mae` | gradient boosting, absolute-error loss - the workhorse |
| `hgb_deep` | deeper, slower variant of the same |
| `hgb_mse` | squared-error twin, kept as evidence for the robust loss |
| `forest` | random forest, decorrelated errors |
| `ridge` | linear reference |

Three fitted weighting schemes were measured and rejected first: least squares in
log space (blend scored worse than its own best member), greedy selection on
dollar MAE (gave ridge 74% of the weight on the earliest fold, where ridge scored
$498 MAE), and bagged greedy selection (moved the weights under a percentage
point, ruling out sampling noise). The cause is that weights are learned from
models fitted on the pre-blend window and then applied to models refitted on more
data, which reorders them. Across all three folds fitted blending never beat the
best single member, so the fitting is gone. See `models.ensemble_weights`.

## What the data says

- **`quote_signal` is a decoy.** Its correlation with realised rate per mile is
  0.05. Its mean sits right next to the rate-per-mile mean, so it reads like a
  price quote; it is not one. It is dropped everywhere.
- **`market_index` only works in aggregate.** Row-level correlation with rate per
  mile is 0.08, daily-mean correlation is 0.58 - it is a daily market reading seen
  through per-row noise. The model uses a daily calendar built from it (daily
  mean, 7- and 28-day rolling means, and the gap between them), which is also what
  gives the December chart rows a market feature they do not ship with.
- **Rate per mile decays with length of haul**, from about $2.74/mi under 300
  miles to $1.90/mi over 2,500.
- **Eight cities appear in validation that never appear in training** (Chicago,
  Charlotte, Laredo, Norfolk, San Diego, Knoxville, Jackson, Allentown). No raw
  city identifier is used as a feature; geography enters through coordinates,
  great-circle distance and bearing, and every lane encoding ships a `seen_` flag
  with a global fallback.
- **No time index is used as a feature.** A tree cannot split past the last date
  it saw, and the whole task is two months beyond the end of training. Seasonality
  enters only through cyclical calendar terms, holiday proximity and the market
  calendar - all of which are defined for future dates.

### Data quality

| issue | rows | handling |
|---|---|---|
| Negative `weight` (correct magnitude, flipped sign) | 292 | absolute value |
| Missing `weight` | 300 | left missing, flagged; the boosters split on missing natively |
| Missing `market_index` | 374 | filled from that day's market calendar, flagged |
| Injected multiplicative rate shocks | ~1% | not deleted; the primary learner optimises absolute error, which estimates the conditional median and is not dragged by them |
| `distance` long against great-circle distance | 214 | **not** repaired - the ratio is constant within a lane (std 0.02), so it is distortion baked into the synthetic coordinates, not per-row corruption |

## Layout

```
src/data.py       loading, cleaning, city table, market calendar
src/features.py   feature engineering and the leak-free lane encoder
src/models.py     model zoo, greedy blending, calibration, metrics
src/pipeline.py   cleaned frames -> fitted stack -> dollars
src/validate.py   rolling-origin validation and baselines
src/train.py      fits and saves the shipped model
src/predict.py    writes the two submission files
src/eda.py        the numbers and figures quoted in the report
src/selfcheck.py  runnable assertions for the failures that would be silent
src/report.py     the short DOCX report
src/report_pdf.py the long-form PDF write-up
```
