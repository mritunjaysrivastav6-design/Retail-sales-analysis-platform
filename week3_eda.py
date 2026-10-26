"""
WEEK 3: Exploratory Data Analysis
Generates charts on sales trends, seasonality, category/region performance,
and customer distribution.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
CHART_DIR = "/home/claude/retail_project/outputs/charts"

df = pd.read_csv("/home/claude/retail_project/data/transactions.csv", parse_dates=["order_date"])
customers = pd.read_csv("/home/claude/retail_project/data/customers.csv")

# 1. Daily revenue trend with 30-day rolling average
daily = df.groupby("order_date")["net_sales"].sum().reset_index()
daily["rolling_30"] = daily["net_sales"].rolling(30).mean()

plt.figure(figsize=(11, 4.5))
plt.plot(daily.order_date, daily.net_sales, alpha=0.3, label="Daily revenue")
plt.plot(daily.order_date, daily.rolling_30, color="crimson", linewidth=2, label="30-day rolling avg")
plt.title("Daily Revenue Trend (2024–2025)")
plt.xlabel("Date"); plt.ylabel("Revenue")
plt.legend()
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/01_daily_revenue_trend.png", dpi=120)
plt.close()

# 2. Monthly seasonality (avg revenue per calendar month across years)
df["month"] = df.order_date.dt.month
monthly_season = df.groupby("month")["net_sales"].sum().reset_index()
plt.figure(figsize=(8, 4.5))
sns.barplot(data=monthly_season, x="month", y="net_sales", color="steelblue")
plt.title("Total Revenue by Calendar Month (Seasonality)")
plt.xlabel("Month"); plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/02_monthly_seasonality.png", dpi=120)
plt.close()

# 3. Revenue by region
region_rev = df.groupby("region")["net_sales"].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(7, 4.5))
sns.barplot(data=region_rev, x="region", y="net_sales", color="seagreen")
plt.title("Total Revenue by Region")
plt.xlabel("Region"); plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/03_revenue_by_region.png", dpi=120)
plt.close()

# 4. Revenue by category
cat_rev = df.groupby("category")["net_sales"].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(7, 4.5))
sns.barplot(data=cat_rev, x="net_sales", y="category", color="darkorange")
plt.title("Total Revenue by Product Category")
plt.xlabel("Revenue"); plt.ylabel("Category")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/04_revenue_by_category.png", dpi=120)
plt.close()

# 5. Weekday vs weekend order pattern
df["weekday"] = df.order_date.dt.day_name()
weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
wk = df.groupby("weekday")["transaction_id"].count().reindex(weekday_order).reset_index()
plt.figure(figsize=(8, 4.5))
sns.barplot(data=wk, x="weekday", y="transaction_id", color="mediumpurple")
plt.title("Order Volume by Day of Week")
plt.xlabel("Day"); plt.ylabel("Number of Orders")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/05_weekday_pattern.png", dpi=120)
plt.close()

# 6. Customer purchase frequency distribution
freq = df.groupby("customer_id")["transaction_id"].count()
plt.figure(figsize=(8, 4.5))
sns.histplot(freq, bins=30, color="teal")
plt.title("Customer Purchase Frequency Distribution")
plt.xlabel("Number of Orders per Customer"); plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/06_customer_frequency_dist.png", dpi=120)
plt.close()

# Summary stats printed for the report
print("EDA SUMMARY STATISTICS")
print("-" * 40)
print(f"Total revenue: {df.net_sales.sum():,.2f}")
print(f"Total orders: {df.transaction_id.nunique():,}")
print(f"Unique customers who purchased: {df.customer_id.nunique():,}")
print(f"Avg order value: {df.net_sales.mean():,.2f}")
print(f"Best month (by revenue): {monthly_season.loc[monthly_season.net_sales.idxmax(), 'month']}")
print(f"Top region: {region_rev.iloc[0].region} ({region_rev.iloc[0].net_sales:,.0f})")
print(f"Top category: {cat_rev.iloc[0].category} ({cat_rev.iloc[0].net_sales:,.0f})")
print("\n6 charts saved to outputs/charts/")
