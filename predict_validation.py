from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


DATA_DIR = Path(__file__).resolve().parent

TRAIN_PATH = DATA_DIR / "train-test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation-predictions-template.csv"
OUTPUT_PATH = DATA_DIR / "validation_predictions.csv"
MODEL_PATH = DATA_DIR / "freight_rate_model.cbm"


FEATURES = [
    "pickup",
    "delivery",
    "route",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "market_index",
    "quote_signal",
    "quote_total",
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

    frame["quote_total"] = (
        frame["distance"]
        * frame["quote_signal"]
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
train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VALIDATION_PATH)
template = pd.read_csv(TEMPLATE_PATH)

print(f"Training rows: {len(train):,}")
print(f"Validation rows: {len(validation):,}")
print(f"Template rows: {len(template):,}")


# Validate the input files
if len(validation) != 12_000:
    raise ValueError(
        f"Expected 12,000 validation rows, found {len(validation):,}"
    )

if validation["load_id"].duplicated().any():
    raise ValueError("validation.csv contains duplicate load_id values")

if template["load_id"].duplicated().any():
    raise ValueError(
        "The prediction template contains duplicate load_id values"
    )

if set(validation["load_id"]) != set(template["load_id"]):
    raise ValueError(
        "The validation and template load_id values do not match"
    )


# Apply identical feature engineering
train = prepare_features(train)
validation = prepare_features(validation)

X_train = train[FEATURES]
y_train = train["posted_rate"]

X_validation = validation[FEATURES]


# Best iteration 313 means 314 trees
model = CatBoostRegressor(
    iterations=314,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    eval_metric="MAE",
    l2_leaf_reg=5,
    random_seed=42,
    verbose=100,
)


# Retrain using all 48,000 labeled rows
model.fit(
    X_train,
    y_train,
    cat_features=CATEGORICAL_FEATURES,
)


# Save the trained model
model.save_model(MODEL_PATH)


# Predict all validation rows
predicted_rates = model.predict(X_validation)

# The scorer requires positive predictions
predicted_rates = np.clip(
    predicted_rates,
    0.01,
    None,
)


# Associate predictions with IDs
prediction_table = pd.DataFrame(
    {
        "load_id": validation["load_id"],
        "predicted_rate": predicted_rates,
    }
)


# Fill the template by matching load_id
submission = (
    template[["load_id"]]
    .merge(
        prediction_table,
        on="load_id",
        how="left",
        validate="one_to_one",
    )
)


# Final checks
if len(submission) != 12_000:
    raise ValueError("Submission does not contain exactly 12,000 rows")

if submission["predicted_rate"].isna().any():
    raise ValueError("Some load_id values have no prediction")

if not np.isfinite(submission["predicted_rate"]).all():
    raise ValueError("Predictions contain infinite or invalid values")

if (submission["predicted_rate"] <= 0).any():
    raise ValueError("Predictions must be greater than zero")

if list(submission.columns) != ["load_id", "predicted_rate"]:
    raise ValueError("Submission columns are incorrect")


# Save the required submission
submission.to_csv(
    OUTPUT_PATH,
    index=False,
)

print()
print(f"Created: {OUTPUT_PATH}")
print(f"Saved model: {MODEL_PATH}")
print(submission.head())
print()
print(submission["predicted_rate"].describe())