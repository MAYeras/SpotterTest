from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_DIR = Path(__file__).resolve().parents[1]
CUTOFF = "2025-09-01"

# False = predict posted_rate directly
# True = predict the residual from distance × quote_signal
USE_RESIDUAL = False

data = pd.read_csv(DATA_DIR / "train-test.csv")
data["date"] = pd.to_datetime(data["date"])

# Feature engineering
data["route"] = data["pickup"] + " -> " + data["delivery"]
data["quote_total"] = data["distance"] * data["quote_signal"]

data["month"] = data["date"].dt.month
data["day_of_week"] = data["date"].dt.dayofweek
data["day_of_year"] = data["date"].dt.dayofyear
data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)

data["year_sin"] = np.sin(
    2 * np.pi * data["day_of_year"] / 365.25
)
data["year_cos"] = np.cos(
    2 * np.pi * data["day_of_year"] / 365.25
)
data["week_sin"] = np.sin(
    2 * np.pi * data["day_of_week"] / 7
)
data["week_cos"] = np.cos(
    2 * np.pi * data["day_of_week"] / 7
)

features = [
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

categorical_features = [
    "pickup",
    "delivery",
    "route",
    "equipment",
]

# LightGBM expects categorical columns to use category dtype
for column in categorical_features:
    data[column] = (
        data[column]
        .fillna("Missing")
        .astype("category")
    )

# Historical experiment: train Jan-Aug, validate Sep-Oct
train = data[data["date"] < CUTOFF].copy()
valid = data[data["date"] >= CUTOFF].copy()

X_train = train[features]
X_valid = valid[features]

actual_rates = valid["posted_rate"].to_numpy()

train_baseline = train["quote_total"].to_numpy()
valid_baseline = valid["quote_total"].to_numpy()

if USE_RESIDUAL:
    y_train = (
        train["posted_rate"].to_numpy()
        - train_baseline
    )
    y_valid_for_training = (
        actual_rates
        - valid_baseline
    )
else:
    y_train = train["posted_rate"].to_numpy()
    y_valid_for_training = actual_rates
model = lgb.LGBMRegressor(
    objective="regression",
    metric="mae",
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=30,
    colsample_bytree=0.85,
    reg_lambda=5.0,
    random_state=42,
    verbosity=-1,
)

model.fit(
    X_train,
    y_train,
    categorical_feature=categorical_features,
    eval_set=[(X_valid, y_valid_for_training)],
    eval_metric="mae",
    callbacks=[
        lgb.early_stopping(150),
        lgb.log_evaluation(100),
    ],
)
model_output = model.predict(
    X_valid,
    num_iteration=model.best_iteration_,
)

if USE_RESIDUAL:
    predictions = valid_baseline + model_output
else:
    predictions = model_output

predictions = np.clip(predictions, 0.01, None)
absolute_errors = np.abs(actual_rates - predictions)

mae = mean_absolute_error(actual_rates, predictions)
rmse = mean_squared_error(actual_rates, predictions) ** 0.5
wape = absolute_errors.sum() / np.abs(actual_rates).sum()

baseline_mae = mean_absolute_error(
    actual_rates,
    valid_baseline,
)

print(f"Training rows:   {len(train):,}")
print(f"Validation rows: {len(valid):,}")
print(f"Best iteration:  {model.best_iteration_}")
print(f"Baseline MAE:    ${baseline_mae:,.2f}")
print(f"LightGBM MAE:    ${mae:,.2f}")
print(f"RMSE:            ${rmse:,.2f}")
print(f"WAPE:            {wape:.2%}")

print(f"Median absolute error: {np.median(absolute_errors):,.2f}")
print(f"90th percentile error: {np.quantile(absolute_errors, 0.90):,.2f}")
print(f"95th percentile error: {np.quantile(absolute_errors, 0.95):,.2f}")
print(f"Predictions within $100: {(absolute_errors <= 100).mean():.2%}")
print(f"Predictions within $250: {(absolute_errors <= 250).mean():.2%}")
print(f"Predictions within $500: {(absolute_errors <= 500).mean():.2%}")