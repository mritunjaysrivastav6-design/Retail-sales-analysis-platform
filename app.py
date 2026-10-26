"""
WEEK 11: Integrated Streamlit Application
Brings together all models built in Weeks 5-10 into a single interactive app:
  - Tab 1: Business Overview (KPI dashboard - Power BI style)
  - Tab 2: Demand Forecast (actual vs predicted, next-day prediction)
  - Tab 3: Customer Insights (segment, churn risk + SHAP explanation, recommendations)

Run locally with:  streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = f"{BASE}/data"
MODELS = f"{BASE}/models"

st.set_page_config(page_title="Retail Sales & Customer Intelligence Platform",
                    layout="wide", page_icon="📊")

# ---------------------------------------------------------
# Cached data / model loading
# ---------------------------------------------------------
@st.cache_data
def load_data():
    txn = pd.read_csv(f"{DATA}/transactions.csv", parse_dates=["order_date"])
    ts = pd.read_csv(f"{DATA}/features_timeseries.csv", parse_dates=["order_date"])
    rfm = pd.read_csv(f"{DATA}/features_customer_rfm_segmented.csv")
    products = pd.read_csv(f"{DATA}/products.csv")
    return txn, ts, rfm, products

@st.cache_resource
def load_models():
    return {
        "forecast_rf": joblib.load(f"{MODELS}/forecasting_random_forest.pkl"),
        "forecast_xgb": joblib.load(f"{MODELS}/forecasting_xgboost.pkl"),
        "churn_rf": joblib.load(f"{MODELS}/churn_random_forest.pkl"),
        "churn_scaler": joblib.load(f"{MODELS}/churn_scaler.pkl"),
        "kmeans": joblib.load(f"{MODELS}/kmeans_segmentation.pkl"),
        "seg_scaler": joblib.load(f"{MODELS}/segmentation_scaler.pkl"),
        "rec_engine": joblib.load(f"{MODELS}/recommendation_engine.pkl"),
    }

txn, ts, rfm, products = load_data()
models = load_models()

CHURN_FEATURES = ["frequency", "monetary", "avg_order_value", "avg_discount",
                   "tenure_days", "distinct_categories"]
SEGMENT_LABELS = {}
_seg_order = rfm.groupby("segment")["monetary"].mean().sort_values(ascending=False).index.tolist()
_labels_pool = ["High-Value Loyal", "Steady Regular", "Occasional Buyer", "At-Risk / Low-Engagement"]
for i, s in enumerate(_seg_order):
    SEGMENT_LABELS[s] = _labels_pool[i] if i < len(_labels_pool) else f"Segment {s}"

st.title("📊 AI/ML-Powered Retail Sales & Customer Intelligence Platform")
st.caption("End-to-end demo: demand forecasting, churn prediction, customer segmentation, "
           "explainability, and product recommendations.")

tab1, tab2, tab3 = st.tabs(["🏠 Business Overview", "📈 Demand Forecast", "🧑‍🤝‍🧑 Customer Insights"])

# ============================================================
# TAB 1: BUSINESS OVERVIEW (dashboard)
# ============================================================
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"₹{txn.net_sales.sum()/1e7:.1f} Cr")
    col2.metric("Total Orders", f"{txn.transaction_id.nunique():,}")
    col3.metric("Active Customers", f"{rfm.shape[0]:,}")
    col4.metric("Churn Rate", f"{rfm.churned.mean()*100:.1f}%")

    c1, c2 = st.columns(2)
    with c1:
        daily = txn.groupby("order_date")["net_sales"].sum().reset_index()
        fig = px.line(daily, x="order_date", y="net_sales", title="Daily Revenue Trend")
        st.plotly_chart(fig, use_container_width=True)

        region_rev = txn.groupby("region")["net_sales"].sum().reset_index().sort_values("net_sales", ascending=False)
        fig2 = px.bar(region_rev, x="region", y="net_sales", title="Revenue by Region")
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        cat_rev = txn.groupby("category")["net_sales"].sum().reset_index().sort_values("net_sales", ascending=False)
        fig3 = px.bar(cat_rev, x="net_sales", y="category", orientation="h", title="Revenue by Category")
        st.plotly_chart(fig3, use_container_width=True)

        seg_counts = rfm["segment"].map(SEGMENT_LABELS).value_counts().reset_index()
        seg_counts.columns = ["segment", "count"]
        fig4 = px.pie(seg_counts, names="segment", values="count", title="Customer Segment Mix")
        st.plotly_chart(fig4, use_container_width=True)

# ============================================================
# TAB 2: DEMAND FORECAST
# ============================================================
with tab2:
    st.subheader("Forecast Model Performance (last 60 days held out)")
    perf = pd.read_csv(f"{BASE}/outputs/forecasting_model_comparison.csv", index_col=0)
    st.dataframe(perf, use_container_width=True)

    st.subheader("Predict Next-Day Revenue")
    st.write("Adjust recent sales momentum to see how the model's prediction responds:")

    latest = ts.iloc[-1]
    col1, col2, col3 = st.columns(3)
    lag_1 = col1.number_input("Yesterday's revenue", value=float(latest["lag_1"]))
    rolling_mean_7 = col2.number_input("7-day rolling average", value=float(latest["rolling_mean_7"]))
    is_festive = col3.selectbox("Festive season?", ["No", "Yes"], index=int(latest["is_festive_season"]))

    input_row = pd.DataFrame([{
        "day_of_week": latest["day_of_week"], "month": latest["month"],
        "is_weekend": latest["is_weekend"],
        "is_festive_season": 1 if is_festive == "Yes" else 0,
        "lag_1": lag_1, "lag_7": latest["lag_7"], "lag_14": latest["lag_14"],
        "rolling_mean_7": rolling_mean_7, "rolling_std_7": latest["rolling_std_7"],
        "rolling_mean_30": latest["rolling_mean_30"], "rolling_std_30": latest["rolling_std_30"],
    }])
    pred = models["forecast_rf"].predict(input_row)[0]
    st.metric("Predicted Next-Day Revenue", f"₹{pred:,.0f}")

# ============================================================
# TAB 3: CUSTOMER INSIGHTS
# ============================================================
with tab3:
    customer_id = st.selectbox("Select a customer", rfm["customer_id"].tolist())
    cust_row = rfm[rfm["customer_id"] == customer_id].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Segment", SEGMENT_LABELS[cust_row["segment"]])
    col2.metric("Frequency (orders)", int(cust_row["frequency"]))
    col3.metric("Lifetime Value", f"₹{cust_row['monetary']:,.0f}")

    # Churn prediction + SHAP explanation
    X_cust = pd.DataFrame([cust_row[CHURN_FEATURES].values], columns=CHURN_FEATURES)
    X_cust_scaled = pd.DataFrame(models["churn_scaler"].transform(X_cust), columns=CHURN_FEATURES)
    churn_proba = models["churn_rf"].predict_proba(X_cust_scaled)[0, 1]

    st.subheader("Churn Risk")
    risk_color = "🔴 High" if churn_proba > 0.5 else ("🟡 Medium" if churn_proba > 0.25 else "🟢 Low")
    st.metric("Predicted Churn Probability", f"{churn_proba*100:.1f}%", risk_color)

    explainer = shap.TreeExplainer(models["churn_rf"])
    shap_vals = explainer.shap_values(X_cust_scaled)
    sv = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0, :, 1]
    driver_df = pd.DataFrame({"feature": CHURN_FEATURES, "impact": sv}).sort_values(
        "impact", key=abs, ascending=False)

    st.write("**Why this prediction (SHAP):**")
    for _, row in driver_df.head(3).iterrows():
        direction = "increases" if row["impact"] > 0 else "decreases"
        st.write(f"- `{row['feature']}` {direction} churn risk "
                 f"(impact: {row['impact']:+.3f})")

    # Recommendations
    st.subheader("Recommended Products")
    rec = models["rec_engine"]
    if customer_id in rec["cust_idx"]:
        c_i = rec["cust_idx"][customer_id]
        scores = (rec["customer_factors"][c_i] @ rec["product_factors"].T).copy()
        already_bought = rec["matrix"][c_i].nonzero()[1]
        scores[already_bought] = -np.inf
        top_idx = np.argsort(scores)[::-1][:5]
        idx_to_prod = {v: k for k, v in rec["prod_idx"].items()}
        rec_products = [idx_to_prod[i] for i in top_idx]
    else:
        rec_products = rec["top_popular_products"][:5]

    cat_map = dict(zip(products.product_id, products.category))
    price_map = dict(zip(products.product_id, products.unit_price))
    rec_df = pd.DataFrame({
        "Product": rec_products,
        "Category": [cat_map.get(p, "?") for p in rec_products],
        "Price": [f"₹{price_map.get(p, 0):,.0f}" for p in rec_products],
    })
    st.dataframe(rec_df, use_container_width=True, hide_index=True)
