"""
WEEK 2: SQL Exploration
Runs the SQL analysis queries against the SQLite database and prints results.
"""
import sqlite3
import pandas as pd

conn = sqlite3.connect("/home/claude/retail_project/data/retail.db")

print("=" * 60)
print("1. Monthly Revenue Trend (first 6 months)")
print("=" * 60)
q1 = """
SELECT strftime('%Y-%m', order_date) AS month,
       ROUND(SUM(net_sales), 2) AS total_revenue,
       COUNT(DISTINCT transaction_id) AS num_orders
FROM transactions GROUP BY month ORDER BY month
"""
df1 = pd.read_sql(q1, conn)
print(df1.head(6).to_string(index=False))

print("\n" + "=" * 60)
print("2. Revenue by Region x Category (Top 10)")
print("=" * 60)
q2 = """
SELECT t.region, p.category, ROUND(SUM(t.net_sales), 2) AS revenue
FROM transactions t JOIN products p ON t.product_id = p.product_id
GROUP BY t.region, p.category ORDER BY revenue DESC LIMIT 10
"""
print(pd.read_sql(q2, conn).to_string(index=False))

print("\n" + "=" * 60)
print("3. Top 10 Customers by Lifetime Value (RANK window function)")
print("=" * 60)
q3 = """
SELECT customer_id, ROUND(SUM(net_sales), 2) AS lifetime_value,
       RANK() OVER (ORDER BY SUM(net_sales) DESC) AS ltv_rank
FROM transactions GROUP BY customer_id ORDER BY ltv_rank LIMIT 10
"""
print(pd.read_sql(q3, conn).to_string(index=False))

print("\n" + "=" * 60)
print("4. Month-over-Month Growth (CTE + LAG)")
print("=" * 60)
q4 = """
WITH monthly AS (
    SELECT strftime('%Y-%m', order_date) AS month, SUM(net_sales) AS revenue
    FROM transactions GROUP BY month
)
SELECT month, ROUND(revenue,2) AS revenue,
       ROUND(revenue - LAG(revenue) OVER (ORDER BY month), 2) AS mom_change,
       ROUND(100.0*(revenue - LAG(revenue) OVER (ORDER BY month))/LAG(revenue) OVER (ORDER BY month), 2) AS mom_growth_pct
FROM monthly ORDER BY month
"""
df4 = pd.read_sql(q4, conn)
print(df4.head(6).to_string(index=False))
df4.to_csv("/home/claude/retail_project/data/monthly_revenue_growth.csv", index=False)

print("\n" + "=" * 60)
print("5. RFM Base Table (Top 10 by monetary)")
print("=" * 60)
q5 = """
WITH last_order AS (
    SELECT customer_id, MAX(order_date) AS last_purchase_date
    FROM transactions GROUP BY customer_id
),
rfm AS (
    SELECT t.customer_id, COUNT(DISTINCT t.transaction_id) AS frequency,
           ROUND(SUM(t.net_sales), 2) AS monetary, l.last_purchase_date
    FROM transactions t JOIN last_order l ON t.customer_id = l.customer_id
    GROUP BY t.customer_id
)
SELECT * FROM rfm ORDER BY monetary DESC LIMIT 10
"""
print(pd.read_sql(q5, conn).to_string(index=False))

conn.close()
print("\nSQL exploration complete. Monthly revenue growth saved to data/monthly_revenue_growth.csv")
