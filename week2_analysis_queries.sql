-- WEEK 2: SQL Exploration
-- Demonstrates joins, aggregations, window functions, and CTEs on the retail database.

-- 1. Monthly revenue trend
SELECT strftime('%Y-%m', order_date) AS month,
       ROUND(SUM(net_sales), 2)      AS total_revenue,
       COUNT(DISTINCT transaction_id) AS num_orders
FROM transactions
GROUP BY month
ORDER BY month;

-- 2. Revenue by region and category (JOIN not needed here since transactions
--    already carries denormalized region/category, but shown for products table)
SELECT t.region, p.category, ROUND(SUM(t.net_sales), 2) AS revenue
FROM transactions t
JOIN products p ON t.product_id = p.product_id
GROUP BY t.region, p.category
ORDER BY revenue DESC
LIMIT 10;

-- 3. Top 10 customers by lifetime value, with rank (WINDOW FUNCTION)
SELECT customer_id,
       ROUND(SUM(net_sales), 2) AS lifetime_value,
       RANK() OVER (ORDER BY SUM(net_sales) DESC) AS ltv_rank
FROM transactions
GROUP BY customer_id
ORDER BY ltv_rank
LIMIT 10;

-- 4. Month-over-month revenue growth (CTE + window LAG function)
WITH monthly AS (
    SELECT strftime('%Y-%m', order_date) AS month,
           SUM(net_sales) AS revenue
    FROM transactions
    GROUP BY month
)
SELECT month,
       ROUND(revenue, 2) AS revenue,
       ROUND(revenue - LAG(revenue) OVER (ORDER BY month), 2) AS mom_change,
       ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
             / LAG(revenue) OVER (ORDER BY month), 2) AS mom_growth_pct
FROM monthly
ORDER BY month;

-- 5. Customer recency, frequency, monetary (RFM) base table (CTE)
WITH last_order AS (
    SELECT customer_id, MAX(order_date) AS last_purchase_date
    FROM transactions
    GROUP BY customer_id
),
rfm AS (
    SELECT t.customer_id,
           COUNT(DISTINCT t.transaction_id) AS frequency,
           ROUND(SUM(t.net_sales), 2)       AS monetary,
           l.last_purchase_date
    FROM transactions t
    JOIN last_order l ON t.customer_id = l.customer_id
    GROUP BY t.customer_id
)
SELECT * FROM rfm ORDER BY monetary DESC LIMIT 10;

-- 6. Running (cumulative) revenue total per month, using window SUM
WITH monthly AS (
    SELECT strftime('%Y-%m', order_date) AS month, SUM(net_sales) AS revenue
    FROM transactions GROUP BY month
)
SELECT month, ROUND(revenue,2) AS revenue,
       ROUND(SUM(revenue) OVER (ORDER BY month), 2) AS cumulative_revenue
FROM monthly ORDER BY month;
