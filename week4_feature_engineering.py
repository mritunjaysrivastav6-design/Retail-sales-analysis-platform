"""
WEEK 4: Feature Engineering
Builds:
  1. Daily sales time-series features (lags, rolling averages, calendar features)
     -> used later for demand forecasting
  2. Customer-level RFM + behavioral features
     -> used later for churn prediction and segmentation
"""
import pandas as pd
import numpy as np

df = pd.read_csv("/home/claude/retail_project/data/transactions.csv", parse_dates=["order_date"])

# ---------------------------------------------------------
# 1. TIME-SERIES FEATURES (daily aggregated sales)
# ---------------------------------------------------------
daily = df.groupby("order_date").agg(
    daily_revenue=("net_sales", "sum"),
    daily_orders=("transaction_id", "count")
).reset_index().sort_values("order_date")

# Calendar features
daily["day_of_week"] = daily.order_date.dt.dayofweek
daily["month"] = daily.order_date.dt.month
daily["is_weekend"] = daily.day_of_week.isin([5, 6]).astype(int)
daily["is_festive_season"] = daily.month.isin([10, 11, 12]).astype(int)

# Lag features (previous 1, 7, 14 day revenue)
for lag in [1, 7, 14]:
    daily[f"lag_{lag}"] = daily["daily_revenue"].shift(lag)

# Rolling window features (7-day, 30-day mean & std)
for window in [7, 30]:
    daily[f"rolling_mean_{window}"] = daily["daily_revenue"].shift(1).rolling(window).mean()
    daily[f"rolling_std_{window}"] = daily["daily_revenue"].shift(1).rolling(window).std()

daily = daily.dropna().reset_index(drop=True)
daily.to_csv("/home/claude/retail_project/data/features_timeseries.csv", index=False)
print(f"Time-series feature table: {daily.shape[0]} rows, {daily.shape[1]} columns")
print("Columns:", list(daily.columns))

# ---------------------------------------------------------
# 2. CUSTOMER-LEVEL RFM + BEHAVIORAL FEATURES
# ---------------------------------------------------------
snapshot_date = df.order_date.max() + pd.Timedelta(days=1)

rfm = df.groupby("customer_id").agg(
    recency_days=("order_date", lambda x: (snapshot_date - x.max()).days),
    frequency=("transaction_id", "count"),
    monetary=("net_sales", "sum"),
    avg_order_value=("net_sales", "mean"),
    avg_discount=("discount_pct", "mean"),
    tenure_days=("order_date", lambda x: (snapshot_date - x.min()).days),
    distinct_categories=("category", "nunique"),
).reset_index()

# Churn label: a customer is "churned" if their most recent purchase was more
# than 90 days before the snapshot date (a common business rule for this
# purchase cadence). This becomes the target for Week 7's classification model.
rfm["churned"] = (rfm["recency_days"] > 90).astype(int)

rfm.to_csv("/home/claude/retail_project/data/features_customer_rfm.csv", index=False)
print(f"\nCustomer RFM feature table: {rfm.shape[0]} rows, {rfm.shape[1]} columns")
print("Columns:", list(rfm.columns))
print(f"\nChurn rate in dataset: {rfm.churned.mean()*100:.2f}%")
print(rfm.describe().round(2).to_string())
