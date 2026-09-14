from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


DATA_DIR = Path(__file__).resolve().parent

TRAIN_PATH = DATA_DIR / "train-test.csv"
DECEMBER_INPUT_PATH = DATA_DIR / "december_chart_inputs.csv"
DECEMBER_OUTPUT_PATH = DATA_DIR / "december_chart_inputs.csv"
MODEL_PATH = DATA_DIR / "december_freight_model.cbm"


FEATURES = [
    "pickup",
    "delivery",
    "route",
    "distance",
    "equipment",
    "weight",
    "month",
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "year_sin",
    "year_cos",
    "week_sin",
    "week_cos",
]

CATEGORICAL_FEATURES = [
    "pickup",
    "delivery",
    "route",
    "equipment",
]


def prepare_features(frame):
    frame = frame.copy()

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="raise",
    )

    frame["route"] = (
        frame["pickup"]
        + " -> "
        + frame["delivery"]
    )

    frame["month"] = frame["date"].dt.month
    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["day_of_year"] = frame["date"].dt.dayofyear

    frame["week_of_year"] = (
        frame["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    frame["year_sin"] = np.sin(
        2 * np.pi * frame["day_of_year"] / 365.25
    )
    frame["year_cos"] = np.cos(
        2 * np.pi * frame["day_of_year"] / 365.25
    )
    frame["week_sin"] = np.sin(
        2 * np.pi * frame["day_of_week"] / 7
    )
    frame["week_cos"] = np.cos(
        2 * np.pi * frame["day_of_week"] / 7
    )

    for column in CATEGORICAL_FEATURES:
        frame[column] = (
            frame[column]
            .fillna("Missing")
            .astype(str)
        )

    return frame


# Load data
train_raw = pd.read_csv(TRAIN_PATH)
december_raw = pd.read_csv(DECEMBER_INPUT_PATH)

train = prepare_features(train_raw)
december = prepare_features(december_raw)


# Use September–October to select the tree count
cutoff = pd.Timestamp("2025-09-01")

development_train = train[train["date"] < cutoff]
development_valid = train[train["date"] >= cutoff]

selection_model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    eval_metric="MAE",
    l2_leaf_reg=5,
    random_seed=42,
    verbose=100,
)

selection_model.fit(
    development_train[FEATURES],
    development_train["posted_rate"],
    cat_features=CATEGORICAL_FEATURES,
    eval_set=(
        development_valid[FEATURES],
        development_valid["posted_rate"],
    ),
    early_stopping_rounds=150,
    use_best_model=True,
)

best_iterations = selection_model.get_best_iteration() + 1

print(f"Best number of trees: {best_iterations}")


# Retrain the reduced model using all 48,000 rows
final_model = CatBoostRegressor(
    iterations=best_iterations,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    l2_leaf_reg=5,
    random_seed=42,
    verbose=100,
)

final_model.fit(
    train[FEATURES],
    train["posted_rate"],
    cat_features=CATEGORICAL_FEATURES,
)

final_model.save_model(MODEL_PATH)


# Predict all 31 December rows
predicted_rates = final_model.predict(
    december[FEATURES]
)

predicted_rates = np.clip(
    predicted_rates,
    0.01,
    None,
)


# Keep the exact columns and order required by score.py
output = december_raw[
    [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
    ]
].copy()

output["predicted_rate"] = predicted_rates


# Final checks
if len(output) != 31:
    raise ValueError(
        f"Expected 31 December rows, found {len(output)}"
    )

if output["predicted_rate"].isna().any():
    raise ValueError("Some December predictions are missing")

if not np.isfinite(output["predicted_rate"]).all():
    raise ValueError("December predictions contain invalid values")

if (output["predicted_rate"] <= 0).any():
    raise ValueError("All predicted rates must be positive")


output.to_csv(
    DECEMBER_OUTPUT_PATH,
    index=False,
)

print(f"Created: {DECEMBER_OUTPUT_PATH}")
print(output.to_string(index=False))