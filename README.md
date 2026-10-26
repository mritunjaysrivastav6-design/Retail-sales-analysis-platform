# AI/ML-Powered Retail Sales & Customer Intelligence Platform

A 12-week college project combining SQL data engineering, exploratory analysis,
and three machine learning paradigms (regression, classification, clustering)
into one deployable retail intelligence platform — built for a B.Tech AI
placement portfolio.

## What it does

1. **Demand Forecasting** — predicts next-day revenue using Random Forest /
   XGBoost regressors trained on engineered time-series features, benchmarked
   against an ARIMA statistical baseline.
2. **Customer Churn Prediction** — classifies customers as likely to churn
   using Logistic Regression and Random Forest, with SMOTE to handle class
   imbalance.
3. **Customer Segmentation** — groups customers into four business-actionable
   segments (High-Value Loyal, Steady Regular, Occasional Buyer, At-Risk)
   using K-Means clustering on RFM features.
4. **Explainability** — SHAP values explain individual forecast and churn
   predictions in plain language.
5. **Recommendation Engine** — collaborative filtering (matrix factorization)
   recommends products per customer, with a popularity-based cold-start
   fallback for new customers.
6. **Deployment** — a Streamlit app ties every model into one interactive
   tool; an HTML business dashboard gives a Power-BI-style overview.

## Project structure

```
retail_project/
├── data/                          # Generated dataset + engineered features
│   ├── customers.csv, products.csv, transactions.csv
│   ├── retail.db                  # SQLite database
│   ├── features_timeseries.csv
│   └── features_customer_rfm_segmented.csv
├── sql/
│   └── week2_analysis_queries.sql # Joins, window functions, CTEs
├── notebooks/                     # One script per week, run in order
│   ├── week1_data_generation.py
│   ├── week2_sql_exploration.py
│   ├── week3_eda.py
│   ├── week4_feature_engineering.py
│   ├── week5_forecasting_models.py
│   ├── week6_customer_segmentation.py
│   ├── week7_churn_classification.py
│   ├── week8_model_validation_summary.py
│   ├── week9_shap_explainability.py
│   └── week10_recommendation_engine.py
├── models/                        # Trained model artifacts (.pkl)
├── outputs/
│   ├── charts/                    # All generated charts (18 PNGs)
│   ├── *_model_comparison.csv     # Evaluation tables
│   └── dashboard_data.json
└── app/
    ├── app.py                     # Streamlit application
    └── dashboard_final.html       # Standalone business dashboard
```

## How to run it

```bash
pip install -r requirements.txt

# Run the pipeline in order (each step reads the previous step's output)
cd notebooks
python week1_data_generation.py
python week2_sql_exploration.py
python week3_eda.py
python week4_feature_engineering.py
python week5_forecasting_models.py
python week6_customer_segmentation.py
python week7_churn_classification.py
python week8_model_validation_summary.py
python week9_shap_explainability.py
python week10_recommendation_engine.py

# Launch the app
cd ../app
streamlit run app.py
```

Open the printed local URL (usually `http://localhost:8501`) in a browser.

## Key results

| Task | Model | Metric | Result |
|---|---|---|---|
| Demand Forecasting | Random Forest | RMSE | 272,587 (29% better than ARIMA baseline) |
| Churn Prediction | Logistic Regression | F1 / ROC-AUC | 0.506 / 0.767 (5-fold CV: 0.525 ± 0.043) |
| Segmentation | K-Means (K=4) | Silhouette Score | 0.309 |

## Importing into Power BI (optional)

The `outputs/*.csv` files (`monthly_revenue_growth.csv`, `segment_profiles.csv`,
and `data/transactions.csv`) are already in a clean, flat format ready for
**Get Data → Text/CSV** import in Power BI Desktop. Recommended visuals:
a line chart for monthly revenue, a matrix for region × category, and a
donut chart for segment mix — mirroring the layout of `dashboard_final.html`.

## Tech stack

Python · pandas · scikit-learn · XGBoost · statsmodels (ARIMA) ·
imbalanced-learn (SMOTE) · SHAP · Streamlit · Plotly · SQLite/SQL

## Talking points for interviews

- Why engineered lag/rolling features let tree-based ML beat a classical
  ARIMA baseline for this data.
- Why Logistic Regression outperformed Random Forest on the churn task after
  SMOTE balancing (simpler model generalized better; RF showed signs of
  overfitting) — validated with 5-fold cross-validation, not a single split.
- Why K=4 was chosen for segmentation over the statistically "optimal" K=2 —
  a business-usefulness trade-off, not just a silhouette-score maximization.
- How SHAP turns a black-box prediction into a specific, defensible business
  explanation for both the forecast and the churn model.
- How the recommendation engine handles the cold-start problem for new
  customers with no purchase history.
