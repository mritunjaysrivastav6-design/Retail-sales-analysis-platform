"""
WEEK 8: Model Validation & Consolidation
Pulls together the evaluation results from all models built so far
(forecasting, segmentation, churn) into one consolidated summary table,
and performs a 5-fold cross-validation sanity check on the churn classifier
to confirm the held-out test performance is not a fluke of the split.
"""
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

forecast_results = pd.read_csv("/home/claude/retail_project/outputs/forecasting_model_comparison.csv", index_col=0)
churn_results = pd.read_csv("/home/claude/retail_project/outputs/churn_model_comparison.csv", index_col=0)
segment_profile = pd.read_csv("/home/claude/retail_project/outputs/segment_profiles.csv")

print("=" * 60)
print("CONSOLIDATED MODEL RESULTS - WEEKS 5, 6, 7")
print("=" * 60)
print("\n[Forecasting] ARIMA baseline vs Random Forest vs XGBoost")
print(forecast_results.to_string())

print("\n[Churn Prediction] Logistic Regression vs Random Forest")
print(churn_results.to_string())

print("\n[Customer Segmentation] 4-segment profile (K-Means)")
print(segment_profile.round(1).to_string(index=False))

# ---------------------------------------------------------
# Cross-validation sanity check for churn model (best model: Logistic Regression)
# ---------------------------------------------------------
rfm = pd.read_csv("/home/claude/retail_project/data/features_customer_rfm_segmented.csv")
feature_cols = ["frequency", "monetary", "avg_order_value", "avg_discount",
                 "tenure_days", "distinct_categories"]
X = rfm[feature_cols]
y = rfm["churned"]
X_scaled = StandardScaler().fit_transform(X)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
cv_scores = cross_val_score(log_reg, X_scaled, y, cv=cv, scoring="f1")

print("\n" + "=" * 60)
print("5-FOLD CROSS-VALIDATION CHECK (Logistic Regression, churn, F1 score)")
print("=" * 60)
print(f"Fold scores: {[round(s, 3) for s in cv_scores]}")
print(f"Mean F1: {cv_scores.mean():.3f}  |  Std: {cv_scores.std():.3f}")
print("\nInterpretation: consistent F1 across folds confirms the model's")
print("performance is stable and not an artifact of a lucky train/test split.")

with open("/home/claude/retail_project/outputs/week8_validation_summary.txt", "w") as f:
    f.write("WEEK 8 - CONSOLIDATED VALIDATION SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    f.write("Forecasting model comparison:\n" + forecast_results.to_string() + "\n\n")
    f.write("Churn model comparison:\n" + churn_results.to_string() + "\n\n")
    f.write(f"Churn model 5-fold CV F1 scores: {[round(s,3) for s in cv_scores]}\n")
    f.write(f"Mean F1: {cv_scores.mean():.3f} | Std: {cv_scores.std():.3f}\n")

print("\nValidation summary saved to outputs/week8_validation_summary.txt")
