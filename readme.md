# Freight Rate Prediction Challenge

Predict posted freight rates for 12,000 November-December 2025 loads and the supplied fixed December scenario. The implementation, evaluation, and deliverables follow the supplied README and unmodified `score.py`. `Freight_Rate_ML_Assessment.pdf` was not available during this review: confirm its additional requirements before submission.

## Setup

Use Python 3.12. On macOS, LightGBM requires `brew install libomp` (Linux installations may require `libgomp1`).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# For the package versions used in the recorded evaluation:
# python -m pip install -r requirements-tested.txt
```

Place the assessment files in the repository root with these exact names:

- `train-test.csv` (48,000 labeled rows)
- `validation.csv` (12,000 unlabeled rows)
- `validation-predictions-template.csv`
- `december_chart_inputs.csv` (the seven-column, 31-row scenario file)

The first three files are ignored by Git; obtain them from the assessment package. The December file contains completed predictions; rerunning replaces only its prediction values. Original local outputs are preserved under `artifacts/original_*` and excluded from Git.

## Reproduce

```bash
python freight.py all
python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs.csv
python -m unittest discover -s tests -v
```

`evaluate` and `submit` can also be run separately: `python freight.py evaluate`, then `python freight.py submit`. Selection settings are saved in `artifacts/selection.json`. Submission training uses all labeled rows. The compatibility entry points `predict_validation.py` and `predict_december.py` regenerate only their respective outputs using that saved selection.

## Evaluation method

Train January-June and use July-August MAE to select tree count and model family. Refit through August and evaluate September-October without early stopping on those evaluation labels. Compare CatBoost and LightGBM with common features and a quote-total baseline (`distance * quote_signal`) for the full task. For December, compare both models with only available scenario fields and a training-median baseline. The same chronological partitions apply to both tasks.

September-October was used in earlier exploratory work, so this is a retrospective comparison, not an untouched test set. Do not present it as an unbiased final score. The supplied scorer validates output contracts and creates the chart; Spotter computes hidden-label submission metrics.

Features include route, equipment, distance, weight, month, weekday, and annual sine/cosine terms. The full model also uses coordinates, market index, quote signal, and quote total. Tree models handle missing numerical values natively; missing categoricals become `Missing`. LightGBM category mappings are learned only from each training partition; unseen values are treated as missing. IDs and labels are excluded from predictors. Prediction values are floored at $0.01 to satisfy the scorer.

December has no quote signal, market index, or coordinates. It therefore uses a separately evaluated reduced model. Training contains no November/December observations and less than one annual cycle: the scenario chart is a model projection, not evidence of learned December/holiday behavior. Availability of quote and market signals at real quote time needs confirmation from the assessment/data owner.

The selected December model returns $916.24 for each of the 31 dates in this run. Its flat chart reflects the model output; no artificial seasonality was added.

## Files and deliverables

- `freight.py`: shared feature engineering, audit, evaluation, selection, and final training.
- `artifacts/audit.json`, `metrics.csv`, and `selection.json`: reproducible audit and numeric results.
- `artifacts/*_evaluation_predictions.csv`, `*_worst_errors.csv`: row-level evaluation evidence.
- `validation_predictions.csv`: required ID-matched submission.
- `december_chart_inputs.csv`: completed fixed scenario.
- `scorer_results/candidate_december.png`: supplied scorer chart.
- `reports/freight_assessment_report.pdf`: report, including validation and chart.
- `reports/loom_script.md`: 2-3 minute recording script.
- `experiments/`: preserved historical scripts; use the main pipeline for reproduction.

Report generation (optional): install `reportlab>=4,<5`, then run `python reports/build_report.py` after the scorer. The recording must be made by the candidate; add the real Loom link here before submission. Push the finished repository to your GitHub account and verify access for reviewers. Neither a GitHub upload nor Loom recording is performed by the local training pipeline.
