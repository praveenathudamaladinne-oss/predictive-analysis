"""
Predictive Analytics Using Historical Data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
print("=" * 60)
print("STEP 1: Loading Data")
print("=" * 60)

print("Current Folder:", os.getcwd())
print("Files Available:", os.listdir())

df = pd.read_csv("historical_sales.csv", parse_dates=["Date"])

df = df.sort_values("Date").reset_index(drop=True)

print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nMissing Values:\n{df.isnull().sum()}")

# ------------------------------------------------------------
# 2. CLEAN & PREPROCESS
# ------------------------------------------------------------
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
df["Sales"] = df["Sales"].interpolate()
df["Sales"] = df["Sales"].bfill()
df["Sales"] = df["Sales"].ffill()

df = df.drop_duplicates(subset="Date")

df["t"] = np.arange(len(df))
df["month"] = df["Date"].dt.month
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

print("\nData Cleaned Successfully")

# ------------------------------------------------------------
# 3. TRAIN TEST SPLIT
# ------------------------------------------------------------
TEST_SIZE = min(6, max(1, len(df) // 3))

train = df.iloc[:-TEST_SIZE].copy()
test = df.iloc[-TEST_SIZE:].copy()

print("\nTrain Rows:", len(train))
print("Test Rows :", len(test))

# ------------------------------------------------------------
# 4A. LINEAR REGRESSION
# ------------------------------------------------------------
features = ["t", "month_sin", "month_cos"]

X_train = train[features]
y_train = train["Sales"]

X_test = test[features]
y_test = test["Sales"]

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_preds = lr_model.predict(X_test)

# ------------------------------------------------------------
# 4B. EXPONENTIAL SMOOTHING
# ------------------------------------------------------------
def exponential_smoothing(series, alpha=0.3):
    result = [series[0]]

    for n in range(1, len(series)):
        result.append(
            alpha * series[n] +
            (1 - alpha) * result[n - 1]
        )

    return np.array(result)

alpha = 0.3

smoothed_train = exponential_smoothing(
    train["Sales"].values,
    alpha
)

last_level = smoothed_train[-1]

if len(train) > 1:
    trend_step = (
        train["Sales"].iloc[-1] -
        train["Sales"].iloc[0]
    ) / (len(train) - 1)
else:
    trend_step = 0

es_preds = np.array([
    last_level + trend_step * (i + 1)
    for i in range(TEST_SIZE)
])

# ------------------------------------------------------------
# 5. EVALUATION
# ------------------------------------------------------------
def evaluate(y_true, y_pred, model_name):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    mape = np.mean(
        np.abs(
            (y_true - y_pred) /
            np.where(y_true == 0, 1, y_true)
        )
    ) * 100

    print(f"\n{model_name}")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")

    return mae, rmse, mape

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

lr_metrics = evaluate(
    y_test.values,
    lr_preds,
    "Linear Regression"
)

es_metrics = evaluate(
    y_test.values,
    es_preds,
    "Exponential Smoothing"
)

best_model = (
    "Linear Regression"
    if lr_metrics[1] < es_metrics[1]
    else "Exponential Smoothing"
)

print("\nBest Model:", best_model)

# ------------------------------------------------------------
# 6. FUTURE FORECAST
# ------------------------------------------------------------
future_periods = 6

last_date = df["Date"].iloc[-1]

future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=future_periods,
    freq="MS"
)

future_t = np.arange(
    len(df),
    len(df) + future_periods
)

future_month = future_dates.month

future_df = pd.DataFrame({
    "t": future_t,
    "month_sin": np.sin(
        2 * np.pi * future_month / 12
    ),
    "month_cos": np.cos(
        2 * np.pi * future_month / 12
    )
})

lr_full = LinearRegression()

lr_full.fit(
    df[features],
    df["Sales"]
)

future_forecast = lr_full.predict(
    future_df
)

forecast_table = pd.DataFrame({
    "Date": future_dates,
    "Forecasted_Sales": np.round(
        future_forecast,
        2
    )
})

print("\nFuture Forecast")
print(forecast_table)

forecast_table.to_csv(
    "future_forecast.csv",
    index=False
)

# ------------------------------------------------------------
# 7. VISUALIZATION
# ------------------------------------------------------------
fig, axes = plt.subplots(
    2,
    1,
    figsize=(12, 10)
)

# Actual vs Predicted
axes[0].plot(
    df["Date"],
    df["Sales"],
    label="Actual"
)

axes[0].plot(
    test["Date"],
    lr_preds,
    "--o",
    label="Linear Regression"
)

axes[0].plot(
    test["Date"],
    es_preds,
    "--s",
    label="Exp Smoothing"
)

axes[0].axvline(
    train["Date"].iloc[-1],
    linestyle=":"
)

axes[0].set_title(
    "Actual vs Predicted"
)

axes[0].legend()
axes[0].grid(True)

# Forecast Plot
axes[1].plot(
    df["Date"],
    df["Sales"],
    label="Historical Sales"
)

axes[1].plot(
    forecast_table["Date"],
    forecast_table["Forecasted_Sales"],
    "--o",
    label="Future Forecast"
)

axes[1].axvline(
    df["Date"].iloc[-1],
    linestyle=":"
)

axes[1].set_title(
    "Future Forecast"
)

axes[1].legend()
axes[1].grid(True)

plt.tight_layout()

plt.savefig(
    "forecast_visualization.png",
    dpi=150
)

plt.show()

# ------------------------------------------------------------
# 8. SAVE METRICS
# ------------------------------------------------------------
metrics_summary = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Exponential Smoothing"
    ],
    "MAE": [
        lr_metrics[0],
        es_metrics[0]
    ],
    "RMSE": [
        lr_metrics[1],
        es_metrics[1]
    ],
    "MAPE (%)": [
        lr_metrics[2],
        es_metrics[2]
    ]
})

metrics_summary.to_csv(
    "model_metrics.csv",
    index=False
)

print("\nFiles Generated Successfully:")
print("future_forecast.csv")
print("model_metrics.csv")
print("forecast_visualization.png")