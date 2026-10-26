"""
WEEK 5: Demand Forecasting
Trains an ARIMA baseline and compares it against Random Forest / XGBoost
regressors that use the engineered lag/rolling/calendar features.
Evaluates all models with RMSE and MAE on a held-out (last 60 days) test set.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

CHART_DIR = "/home/claude/retail_project/outputs/charts"
df = pd.read_csv("/home/claude/retail_project/data/features_timeseries.csv", parse_dates=["order_date"])
df = df.sort_values("order_date").reset_index(drop=True)

TEST_DAYS = 60
train = df.iloc[:-TEST_DAYS]
test = df.iloc[-TEST_DAYS:]

feature_cols = ["day_of_week", "month", "is_weekend", "is_festive_season",
                 "lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_std_7",
                 "rolling_mean_30", "rolling_std_30"]
target_col = "daily_revenue"

X_train, y_train = train[feature_cols], train[target_col]
X_test, y_test = test[feature_cols], test[target_col]

results = {}

# ---------------------------------------------------------
# 1. ARIMA baseline (classical time-series statistics)
# ---------------------------------------------------------
from statsmodels.tsa.arima.model import ARIMA

arima_train_series = train.set_index("order_date")[target_col]
arima_model = ARIMA(arima_train_series, order=(5, 1, 2)).fit()
arima_forecast = arima_model.forecast(steps=TEST_DAYS)
arima_forecast.index = test.order_date.values

rmse_arima = np.sqrt(mean_squared_error(y_test, arima_forecast))
mae_arima = mean_absolute_error(y_test, arima_forecast)
results["ARIMA (baseline)"] = {"RMSE": rmse_arima, "MAE": mae_arima}

# ---------------------------------------------------------
# 2. Random Forest Regressor
# ---------------------------------------------------------
rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, rf_pred))
mae_rf = mean_absolute_error(y_test, rf_pred)
results["Random Forest"] = {"RMSE": rmse_rf, "MAE": mae_rf}

# ---------------------------------------------------------
# 3. XGBoost Regressor
# ---------------------------------------------------------
xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                              subsample=0.9, colsample_bytree=0.9, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, xgb_pred))
mae_xgb = mean_absolute_error(y_test, xgb_pred)
results["XGBoost"] = {"RMSE": rmse_xgb, "MAE": mae_xgb}

# ---------------------------------------------------------
# Results table
# ---------------------------------------------------------
results_df = pd.DataFrame(results).T.round(2)
results_df["RMSE_%_of_avg_sales"] = (results_df["RMSE"] / y_test.mean() * 100).round(2)
print("FORECASTING MODEL COMPARISON (last 60 days held out)")
print("=" * 60)
print(results_df.to_string())
results_df.to_csv("/home/claude/retail_project/outputs/forecasting_model_comparison.csv")

best_model = results_df["RMSE"].idxmin()
print(f"\nBest performing model: {best_model}")

# ---------------------------------------------------------
# Plot: Actual vs Predicted (best ML model = XGBoost typically)
# ---------------------------------------------------------
plt.figure(figsize=(11, 4.5))
plt.plot(test.order_date, y_test.values, label="Actual", color="black", linewidth=2)
plt.plot(test.order_date, arima_forecast.values, label="ARIMA", linestyle="--", alpha=0.7)
plt.plot(test.order_date, rf_pred, label="Random Forest", linestyle="--", alpha=0.7)
plt.plot(test.order_date, xgb_pred, label="XGBoost", linestyle="--", alpha=0.9, color="crimson")
plt.title("Demand Forecasting: Actual vs Predicted Daily Revenue (Test Period)")
plt.xlabel("Date"); plt.ylabel("Revenue")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/07_forecast_actual_vs_predicted.png", dpi=120)
plt.close()

# Feature importance (XGBoost)
importances = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values()
plt.figure(figsize=(7, 5))
importances.plot(kind="barh", color="teal")
plt.title("XGBoost Feature Importance (Demand Forecasting)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/08_forecast_feature_importance.png", dpi=120)
plt.close()

# Save the trained model + predictions for later use in the app/report
import joblib
joblib.dump(xgb_model, "/home/claude/retail_project/models/forecasting_xgboost.pkl")
joblib.dump(rf, "/home/claude/retail_project/models/forecasting_random_forest.pkl")
print("\nModels saved to models/. Charts saved to outputs/charts/.")
