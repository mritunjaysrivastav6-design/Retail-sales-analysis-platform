"""
WEEK 1: Data Acquisition & Setup
Generates a realistic retail transactions dataset (2 years, seasonality, trend,
multiple regions/categories/customers) and loads it into a SQLite database.
"""
import numpy as np
import pandas as pd
import sqlite3
import os

np.random.seed(42)
OUT_DIR = "/home/claude/retail_project/data"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. Customers
# ---------------------------------------------------------
N_CUSTOMERS = 1200
regions = ["North", "South", "East", "West", "Central"]
segments_seed = ["Consumer", "Corporate", "Home Office"]

customers = pd.DataFrame({
    "customer_id": [f"CUST{i:05d}" for i in range(1, N_CUSTOMERS + 1)],
    "region": np.random.choice(regions, N_CUSTOMERS, p=[0.25, 0.2, 0.2, 0.2, 0.15]),
    "signup_date": pd.to_datetime("2023-01-01") + pd.to_timedelta(
        np.random.randint(0, 500, N_CUSTOMERS), unit="D"),
    "customer_type": np.random.choice(segments_seed, N_CUSTOMERS, p=[0.55, 0.30, 0.15]),
})

# ---------------------------------------------------------
# 2b. Customer behavioral profile (drives realistic purchase patterns
#     and genuine churn, instead of everyone buying at a uniform rate)
# ---------------------------------------------------------
# activity_weight: relative likelihood of being picked for an order each day
# churned_early: a subset of customers who stop purchasing partway through
#                the observation window (creates real churn signal)
customers["activity_weight"] = np.random.gamma(shape=2.0, scale=1.0, size=N_CUSTOMERS)
churn_flag = np.random.rand(N_CUSTOMERS) < 0.12  # ~12% of customers will churn at some point
customers["is_churn_cohort"] = churn_flag
# churn happens at a random point in the last 40% of the 2-year window
window_start = pd.to_datetime("2025-04-01")
window_end = pd.to_datetime("2025-09-30")
random_churn_dates = window_start + pd.to_timedelta(
    np.random.randint(0, (window_end - window_start).days, N_CUSTOMERS), unit="D")
customers["churn_date"] = np.where(churn_flag, random_churn_dates.astype(str), None)

# ---------------------------------------------------------
# 2. Products
# ---------------------------------------------------------
categories = {
    "Electronics": (2000, 45000),
    "Furniture": (1500, 30000),
    "Apparel": (300, 5000),
    "Groceries": (50, 1200),
    "Home & Kitchen": (400, 8000),
}
N_PRODUCTS = 150
prod_rows = []
for i in range(1, N_PRODUCTS + 1):
    cat = np.random.choice(list(categories.keys()))
    lo, hi = categories[cat]
    price = round(np.random.uniform(lo, hi), 2)
    prod_rows.append((f"PROD{i:04d}", cat, price))
products = pd.DataFrame(prod_rows, columns=["product_id", "category", "unit_price"])

# ---------------------------------------------------------
# 3. Transactions (2 years daily, with trend + seasonality + noise)
# ---------------------------------------------------------
start_date = pd.to_datetime("2024-01-01")
end_date = pd.to_datetime("2025-12-31")
date_range = pd.date_range(start_date, end_date, freq="D")

# Base daily order volume with yearly seasonality (festive spike Oct-Dec),
# weekly seasonality (weekend bump) and a gentle upward trend.
t = np.arange(len(date_range))
trend = 1 + t / len(t) * 0.35
yearly_season = 1 + 0.35 * np.sin(2 * np.pi * (date_range.dayofyear / 365) - np.pi / 2)
festive_boost = np.where(date_range.month.isin([10, 11, 12]), 1.4, 1.0)
weekday_boost = np.where(date_range.dayofweek.isin([5, 6]), 1.25, 1.0)
noise = np.random.normal(1, 0.08, len(date_range))

daily_orders = (28 * trend * yearly_season * festive_boost * weekday_boost * noise).astype(int)
daily_orders = np.clip(daily_orders, 5, None)

records = []
cust_ids = customers["customer_id"].values
prod_ids = products["product_id"].values
prod_price_map = dict(zip(products.product_id, products.unit_price))
prod_cat_map = dict(zip(products.product_id, products.category))
cust_region_map = dict(zip(customers.customer_id, customers.region))
cust_type_map = dict(zip(customers.customer_id, customers.customer_type))
cust_weight_map = dict(zip(customers.customer_id, customers.activity_weight))
cust_churn_date_map = dict(zip(customers.customer_id,
                                pd.to_datetime(customers.churn_date, errors="coerce")))

txn_counter = 1
for day, n_orders in zip(date_range, daily_orders):
    # Exclude customers who have already churned by this date, and weight
    # the remaining active customers by their individual activity level.
    churn_dates = np.array([cust_churn_date_map[c] for c in cust_ids])
    active_mask = pd.isna(churn_dates) | (churn_dates > day)
    active_ids = cust_ids[active_mask]
    weights = customers.loc[active_mask, "activity_weight"].values
    weights = weights / weights.sum()

    chosen_customers = np.random.choice(active_ids, size=n_orders, replace=True, p=weights)
    chosen_products = np.random.choice(prod_ids, size=n_orders, replace=True)
    quantities = np.random.randint(1, 6, size=n_orders)
    discount_pcts = np.random.choice([0, 0.05, 0.1, 0.15, 0.2], size=n_orders,
                                      p=[0.5, 0.2, 0.15, 0.1, 0.05])
    for c, p, q, disc in zip(chosen_customers, chosen_products, quantities, discount_pcts):
        unit_price = prod_price_map[p]
        gross = unit_price * q
        net = round(gross * (1 - disc), 2)
        records.append((
            f"TXN{txn_counter:07d}", day.strftime("%Y-%m-%d"), c, p,
            prod_cat_map[p], cust_region_map[c], cust_type_map[c],
            int(q), unit_price, disc, net
        ))
        txn_counter += 1

transactions = pd.DataFrame(records, columns=[
    "transaction_id", "order_date", "customer_id", "product_id", "category",
    "region", "customer_type", "quantity", "unit_price", "discount_pct", "net_sales"
])

print(f"Customers: {len(customers)}")
print(f"Products: {len(products)}")
print(f"Transactions: {len(transactions)}")
print(f"Date range: {transactions.order_date.min()} to {transactions.order_date.max()}")

# ---------------------------------------------------------
# 4. Save CSVs
# ---------------------------------------------------------
customers.to_csv(f"{OUT_DIR}/customers.csv", index=False)
products.to_csv(f"{OUT_DIR}/products.csv", index=False)
transactions.to_csv(f"{OUT_DIR}/transactions.csv", index=False)

# ---------------------------------------------------------
# 5. Load into SQLite (simulates a production SQL database)
# ---------------------------------------------------------
conn = sqlite3.connect(f"{OUT_DIR}/retail.db")
customers.to_sql("customers", conn, if_exists="replace", index=False)
products.to_sql("products", conn, if_exists="replace", index=False)
transactions.to_sql("transactions", conn, if_exists="replace", index=False)
conn.close()

print("\nSQLite database created at data/retail.db with tables: customers, products, transactions")
