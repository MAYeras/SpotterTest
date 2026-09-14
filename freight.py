"""Shared, reproducible training and evaluation for the freight assessment."""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
CAT = ["pickup", "delivery", "route", "equipment"]
REDUCED = CAT + ["distance", "weight", "month", "day_of_week", "year_sin", "year_cos"]
FULL = REDUCED + [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "market_index",
    "quote_signal",
    "quote_total",
]


def features(raw):
    """Create row-local predictors without fitting statistics or modifying inputs."""
    d = raw.copy()
    date = pd.to_datetime(d["date"], errors="raise")
    for c in ["pickup", "delivery", "equipment"]:
        d[c] = d[c].fillna("Missing").astype(str)
    d["route"] = d.pickup + " -> " + d.delivery
    d["month"] = date.dt.month
    d["day_of_week"] = date.dt.dayofweek
    d["year_sin"] = np.sin(2 * np.pi * date.dt.dayofyear / 365.25)
    d["year_cos"] = np.cos(2 * np.pi * date.dt.dayofyear / 365.25)
    if "quote_signal" in d:
        d["quote_total"] = d.distance * d.quote_signal
    return d


def matrix(d, cols, kind, categories=None):
    """Apply training-only categorical levels for LightGBM."""
    x = d[cols].copy()
    if kind == "lightgbm":
        for c in CAT:
            x[c] = pd.Categorical(x[c], categories=categories[c])
    return x


def fit(d, cols, kind, iterations, valid=None):
    """Fit one candidate; early stopping is enabled only for model selection."""
    categories = {c: sorted(d[c].unique()) for c in CAT}
    x = matrix(d, cols, kind, categories)
    if kind == "catboost":
        model = CatBoostRegressor(
            iterations=iterations,
            depth=8,
            learning_rate=0.03,
            loss_function="RMSE",
            eval_metric="MAE",
            l2_leaf_reg=5,
            random_seed=42,
            thread_count=4,
            verbose=False,
            allow_writing_files=False,
        )
        kw = dict(cat_features=CAT)
        if valid is not None:
            kw.update(
                eval_set=(valid[cols], valid.posted_rate), early_stopping_rounds=100
            )
    else:
        model = lgb.LGBMRegressor(
            n_estimators=iterations,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=30,
            colsample_bytree=0.85,
            reg_lambda=5.0,
            random_state=42,
            n_jobs=4,
            verbosity=-1,
        )
        kw = dict(categorical_feature=CAT)
        if valid is not None:
            kw.update(
                eval_set=[(matrix(valid, cols, kind, categories), valid.posted_rate)],
                eval_metric="l1",
                callbacks=[
                    lgb.early_stopping(100, first_metric_only=True, verbose=False)
                ],
            )
            model.set_params(metric="l1")
    model.fit(x, d.posted_rate, **kw)
    return model, categories


def predict(fitted, d, cols, kind):
    model, categories = fitted
    return np.maximum(model.predict(matrix(d, cols, kind, categories)), 0.01)


def metrics(y, p):
    error = np.asarray(p) - np.asarray(y)
    ae = np.abs(error)
    return dict(
        mae=float(ae.mean()),
        rmse=float(np.sqrt(np.mean(error**2))),
        wape=float(ae.sum() / np.abs(y).sum()),
        bias=float(error.mean()),
        p90_absolute_error=float(np.quantile(ae, 0.9)),
        within_250=float((ae <= 250).mean()),
    )


def audit(train, validation, template):
    """Check ID integrity and temporal separation; summarize missing data."""
    for label, d in [
        ("train", train),
        ("validation", validation),
        ("template", template),
    ]:
        if d.load_id.isna().any() or d.load_id.duplicated().any():
            raise ValueError(f"{label}: missing/duplicate IDs")
    if set(train.load_id) & set(validation.load_id):
        raise ValueError("Training and validation IDs overlap")
    if set(validation.load_id) != set(template.load_id):
        raise ValueError("Template IDs do not match validation")
    if train.date.max() >= validation.date.min():
        raise ValueError("Submission dates overlap labeled dates")
    if not np.isfinite(train.posted_rate).all() or (train.posted_rate <= 0).any():
        raise ValueError("Invalid labels")
    common = [c for c in validation if c != "load_id"]
    overlap = train[common].merge(validation[common], on=common).shape[0]
    return dict(
        train_rows=len(train),
        validation_rows=len(validation),
        train_date_range=[train.date.min(), train.date.max()],
        validation_date_range=[validation.date.min(), validation.date.max()],
        missing_train=train.isna().sum().to_dict(),
        missing_validation=validation.isna().sum().to_dict(),
        exact_feature_overlap=overlap,
        duplicate_training_features=int(train.duplicated(common).sum()),
    )


def evaluate():
    """Select on July-August and record retrospective September-October errors."""
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    raw = pd.read_csv(ROOT / "train-test.csv")
    checks = audit(
        raw,
        pd.read_csv(ROOT / "validation.csv"),
        pd.read_csv(ROOT / "validation-predictions-template.csv"),
    )
    d = features(raw)
    early = d[d.date < "2025-07-01"]
    tuning = d[(d.date >= "2025-07-01") & (d.date < "2025-09-01")]
    development = d[d.date < "2025-09-01"]
    test = d[d.date >= "2025-09-01"]
    rows = []
    selected = {}
    predictions = {}
    for mode, cols in [("full", FULL), ("december", REDUCED)]:
        if mode == "full":
            baseline = test.quote_total.to_numpy()
        else:
            baseline = np.full(len(test), development.posted_rate.median())
        rows.append(
            dict(
                mode=mode,
                model="baseline",
                trees=0,
                **metrics(test.posted_rate, baseline),
            )
        )
        predictions[(mode, "baseline")] = baseline
        tuning_scores = {}
        for kind in ["catboost", "lightgbm"]:
            fitted = fit(early, cols, kind, 1500, tuning)
            trees = (
                fitted[0].get_best_iteration() + 1
                if kind == "catboost"
                else fitted[0].best_iteration_
            )
            tuning_scores[kind] = metrics(
                tuning.posted_rate, predict(fitted, tuning, cols, kind)
            )["mae"]
            evaluated = fit(development, cols, kind, trees)
            p = predict(evaluated, test, cols, kind)
            predictions[(mode, kind)] = p
            row = dict(
                mode=mode,
                model=kind,
                trees=int(trees),
                tuning_mae=tuning_scores[kind],
                **metrics(test.posted_rate, p),
            )
            rows.append(row)
            print(row, flush=True)
        # Choose on tuning data, before using retrospective evaluation metrics.
        best = min(tuning_scores, key=tuning_scores.get)
        selected[mode] = next(
            dict(model=r["model"], trees=r["trees"])
            for r in rows
            if r["mode"] == mode and r["model"] == best
        )
        p = predictions[(mode, best)]
        errors = test[
            [
                "load_id",
                "date",
                "pickup",
                "delivery",
                "equipment",
                "distance",
                "posted_rate",
            ]
        ].copy()
        errors["predicted_rate"] = p
        errors["absolute_error"] = np.abs(test.posted_rate - p)
        errors.to_csv(out / f"{mode}_evaluation_predictions.csv", index=False)
        for group in ["equipment", "month"]:
            group_values = test[group]
            for value in sorted(group_values.unique()):
                mask = (group_values == value).to_numpy()
                rows.append(
                    dict(
                        mode=mode,
                        model=best,
                        segment=f"{group}={value}",
                        rows=int(mask.sum()),
                        **metrics(test.posted_rate[mask], p[mask]),
                    )
                )
        errors.nlargest(30, "absolute_error").to_csv(
            out / f"{mode}_worst_errors.csv", index=False
        )
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    (out / "selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    checks["split_rows"] = dict(
        early=len(early),
        tuning=len(tuning),
        development=len(development),
        evaluation=len(test),
    )
    (out / "audit.json").write_text(json.dumps(checks, indent=2) + "\n")
    print("Selected using July-August MAE:", selected, flush=True)


def submit(only=None):
    """Refit selected models on all labels and validate outputs before writing."""
    from score import validate_predictions, validate_december

    train = pd.read_csv(ROOT / "train-test.csv")
    valid = pd.read_csv(ROOT / "validation.csv")
    template = pd.read_csv(ROOT / "validation-predictions-template.csv")
    audit(train, valid, template)
    selected = json.loads((ROOT / "artifacts/selection.json").read_text())
    for mode, cols in [("full", FULL), ("december", REDUCED)]:
        if only is not None and mode != only:
            continue
        choice = selected[mode]
        fitted = fit(features(train), cols, choice["model"], choice["trees"])
        raw = (
            valid if mode == "full" else pd.read_csv(ROOT / "december_chart_inputs.csv")
        )
        p = predict(fitted, features(raw), cols, choice["model"])
        if mode == "full":
            pred = pd.DataFrame({"load_id": raw.load_id, "predicted_rate": p})
            output = template[["load_id"]].merge(
                pred, on="load_id", validate="one_to_one", how="left"
            )
            validate_predictions(output)
            path = ROOT / "validation_predictions.csv"
        else:
            output = raw.copy()
            output["predicted_rate"] = p
            validate_december(output)
            path = ROOT / "december_chart_inputs.csv"
        output.to_csv(path, index=False)
        print("Saved", path, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["evaluate", "submit", "all"])
    args = parser.parse_args()
    if args.command in ["evaluate", "all"]:
        evaluate()
    if args.command in ["submit", "all"]:
        submit()
