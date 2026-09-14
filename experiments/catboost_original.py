from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_DIR = Path(__file__).resolve().parents[1]
TARGET = "posted_rate"

data = pd.read_csv(DATA_DIR / "train-test.csv")
data["date"] = pd.to_datetime(data["date"])

# Basic freight and calendar features
data["route"] = data["pickup"] + " -> " + data["delivery"]
data["quote_total"] = data["distance"] * data["quote_signal"]

data["month"] = data["date"].dt.month
data["day_of_week"] = data["date"].dt.dayofweek
data["day_of_year"] = data["date"].dt.dayofyear
data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)

# Cyclical date features
data["year_sin"] = np.sin(2 * np.pi * data["day_of_year"] / 365.25)
data["year_cos"] = np.cos(2 * np.pi * data["day_of_year"] / 365.25)
data["week_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
data["week_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)

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

# CatBoost categorical columns must contain strings
for column in categorical_features:
    data[column] = data[column].fillna("Missing").astype(str)

# Historical experiment: train Jan-Jul, validate Aug-Oct
train = data[data["date"] < "2025-08-01"].copy()
valid = data[data["date"] >= "2025-08-01"].copy()



X_train = train[features]
y_train = train[TARGET]

X_valid = valid[features]
y_valid = valid[TARGET]
baseline_predictions = (
    valid["distance"] * valid["quote_signal"]
)

baseline_mae = mean_absolute_error(
    y_valid,
    baseline_predictions,
)

model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    eval_metric="MAE",
    l2_leaf_reg=5,
    random_seed=42,
    verbose=100,
)

model.fit(
    X_train,
    y_train,
    cat_features=categorical_features,
    eval_set=(X_valid, y_valid),
    early_stopping_rounds=150,
    use_best_model=True,
)

predictions = model.predict(X_valid)
predictions = np.clip(predictions, 0.01, None)


results = valid[
    [
        "load_id",
        "date",
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "market_index",
        "quote_signal",
        "posted_rate",
    ]
].copy()

results["predicted_rate"] = predictions
results["error"] = results["posted_rate"] - results["predicted_rate"]
results["absolute_error"] = results["error"].abs()

worst_predictions = results.sort_values(
    "absolute_error",
    ascending=False,
).head(30)

print(worst_predictions.to_string(index=False))

worst_predictions.to_csv(
    DATA_DIR / "worst_predictions7.csv",
    index=False,
)

print("Median absolute error:", results["absolute_error"].median())
print("90th percentile error:", results["absolute_error"].quantile(0.90))
print("95th percentile error:", results["absolute_error"].quantile(0.95))

print(
    "Predictions within $100:",
    (results["absolute_error"] <= 100).mean(),
)

print(
    "Predictions within $250:",
    (results["absolute_error"] <= 250).mean(),
)

print(
    "Predictions within $500:",
    (results["absolute_error"] <= 500).mean(),
)


mae = mean_absolute_error(y_valid, predictions)
rmse = mean_squared_error(y_valid, predictions) ** 0.5
wape = np.abs(y_valid - predictions).sum() / np.abs(y_valid).sum()


print(f"Baseline MAE: ${baseline_mae:,.2f}")
print(f"CatBoost MAE: ${mae:,.2f}")

print(f"Training rows:   {len(train):,}")
print(f"Validation rows: {len(valid):,}")
print(f"Best iteration:  {model.get_best_iteration()}")
print(f"MAE:  ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"WAPE: {wape:.2%}")