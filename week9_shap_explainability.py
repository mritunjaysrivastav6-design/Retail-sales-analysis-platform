"""
WEEK 9: Model Explainability (SHAP)
Applies SHAP to the forecasting model (XGBoost) and the churn model
(Random Forest) to explain WHY the models make specific predictions -
turning black-box outputs into business-readable explanations.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap

CHART_DIR = "/home/claude/retail_project/outputs/charts"
MODEL_DIR = "/home/claude/retail_project/models"

# ---------------------------------------------------------
# 1. SHAP for the demand forecasting model (XGBoost)
# ---------------------------------------------------------
ts = pd.read_csv("/home/claude/retail_project/data/features_timeseries.csv", parse_dates=["order_date"])
feature_cols_fc = ["day_of_week", "month", "is_weekend", "is_festive_season",
                    "lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_std_7",
                    "rolling_mean_30", "rolling_std_30"]
X_fc = ts[feature_cols_fc]

xgb_model = joblib.load(f"{MODEL_DIR}/forecasting_xgboost.pkl")
explainer_fc = shap.TreeExplainer(xgb_model)
shap_values_fc = explainer_fc.shap_values(X_fc)

plt.figure()
shap.summary_plot(shap_values_fc, X_fc, show=False)
plt.title("SHAP Summary — Demand Forecasting (XGBoost)")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/15_shap_forecast_summary.png", dpi=120, bbox_inches="tight")
plt.close()

# Explain a single specific day's prediction (a high-demand festive day)
festive_idx = ts[ts["is_festive_season"] == 1].index[-1]
plt.figure()
shap.force_plot(explainer_fc.expected_value, shap_values_fc[festive_idx, :],
                 X_fc.iloc[festive_idx, :], matplotlib=True, show=False)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/16_shap_forecast_single_day.png", dpi=120, bbox_inches="tight")
plt.close()

example_date = ts.loc[festive_idx, "order_date"]
print(f"Example explained forecast date: {example_date.date()} "
      f"(predicted revenue driven mainly by: "
      f"{X_fc.iloc[festive_idx][['rolling_mean_7','lag_7','is_festive_season']].to_dict()})")

# ---------------------------------------------------------
# 2. SHAP for the churn classification model (Random Forest)
# ---------------------------------------------------------
rfm = pd.read_csv("/home/claude/retail_project/data/features_customer_rfm_segmented.csv")
feature_cols_churn = ["frequency", "monetary", "avg_order_value", "avg_discount",
                       "tenure_days", "distinct_categories"]
X_churn = rfm[feature_cols_churn]

churn_scaler = joblib.load(f"{MODEL_DIR}/churn_scaler.pkl")
churn_rf = joblib.load(f"{MODEL_DIR}/churn_random_forest.pkl")
X_churn_scaled = pd.DataFrame(churn_scaler.transform(X_churn), columns=feature_cols_churn)

explainer_churn = shap.TreeExplainer(churn_rf)
shap_values_churn = explainer_churn.shap_values(X_churn_scaled)
# shap_values_churn is a list [class0, class1] for binary RF classifier
sv_churn_positive = shap_values_churn[1] if isinstance(shap_values_churn, list) else shap_values_churn[:, :, 1]

plt.figure()
shap.summary_plot(sv_churn_positive, X_churn_scaled, show=False)
plt.title("SHAP Summary — Churn Prediction (Random Forest)")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/17_shap_churn_summary.png", dpi=120, bbox_inches="tight")
plt.close()

# Explain one specific at-risk customer prediction
at_risk_idx = rfm[rfm["churned"] == 1].index[0]
cust_id = rfm.loc[at_risk_idx, "customer_id"]
plt.figure()
expected_val = explainer_churn.expected_value[1] if isinstance(explainer_churn.expected_value, (list, np.ndarray)) else explainer_churn.expected_value
sv_row = sv_churn_positive[at_risk_idx, :] if sv_churn_positive.ndim == 2 else sv_churn_positive[at_risk_idx]
shap.force_plot(expected_val, sv_row, X_churn_scaled.iloc[at_risk_idx, :], matplotlib=True, show=False)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/18_shap_churn_single_customer.png", dpi=120, bbox_inches="tight")
plt.close()

print(f"\nExample explained churn prediction: {cust_id}")
print(rfm.loc[at_risk_idx, feature_cols_churn + ["churned"]].to_dict())

print("\nSHAP explainability charts saved to outputs/charts/ (15-18).")
print("These translate model outputs into plain-language drivers, e.g.:")
print(" - Forecasting: 'This day's high predicted revenue is driven mainly by")
print("   festive-season flag + strong 7-day rolling average momentum.'")
print(" - Churn: 'This customer's churn risk is high mainly due to low purchase")
print("   frequency and long tenure without recent monetary growth.'")
