"""
WEEK 7: Churn / Risk Prediction (Supervised Classification)
Trains Logistic Regression (baseline) and Random Forest (primary) classifiers
to predict customer churn, handles class imbalance with SMOTE, and evaluates
using precision, recall, F1, and ROC-AUC (accuracy alone is misleading here
since churn is a minority class).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                              roc_curve, precision_score, recall_score, f1_score)
from imblearn.over_sampling import SMOTE
import joblib

CHART_DIR = "/home/claude/retail_project/outputs/charts"
rfm = pd.read_csv("/home/claude/retail_project/data/features_customer_rfm_segmented.csv")

feature_cols = ["frequency", "monetary", "avg_order_value", "avg_discount",
                 "tenure_days", "distinct_categories"]
X = rfm[feature_cols]
y = rfm["churned"]

print(f"Class balance before SMOTE: {y.value_counts().to_dict()} "
      f"({y.mean()*100:.1f}% churned)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# Apply SMOTE only to the training set (never to test data)
# ---------------------------------------------------------
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
print(f"Class balance after SMOTE (train only): {pd.Series(y_train_bal).value_counts().to_dict()}")

results = {}

# ---------------------------------------------------------
# 1. Logistic Regression (baseline, interpretable)
# ---------------------------------------------------------
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_bal, y_train_bal)
lr_pred = log_reg.predict(X_test_scaled)
lr_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

results["Logistic Regression"] = {
    "Precision": precision_score(y_test, lr_pred),
    "Recall": recall_score(y_test, lr_pred),
    "F1": f1_score(y_test, lr_pred),
    "ROC-AUC": roc_auc_score(y_test, lr_proba),
}

# ---------------------------------------------------------
# 2. Random Forest Classifier (primary model, tuned)
# ---------------------------------------------------------
param_grid = {"n_estimators": [200, 300], "max_depth": [5, 8, None]}
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
grid = GridSearchCV(rf_base, param_grid, scoring="f1", cv=3)
grid.fit(X_train_bal, y_train_bal)
rf_clf = grid.best_estimator_
print(f"\nBest Random Forest params: {grid.best_params_}")

rf_pred = rf_clf.predict(X_test_scaled)
rf_proba = rf_clf.predict_proba(X_test_scaled)[:, 1]

results["Random Forest"] = {
    "Precision": precision_score(y_test, rf_pred),
    "Recall": recall_score(y_test, rf_pred),
    "F1": f1_score(y_test, rf_pred),
    "ROC-AUC": roc_auc_score(y_test, rf_proba),
}

results_df = pd.DataFrame(results).T.round(3)
print("\nCHURN MODEL COMPARISON")
print("=" * 60)
print(results_df.to_string())
results_df.to_csv("/home/claude/retail_project/outputs/churn_model_comparison.csv")

# ---------------------------------------------------------
# Confusion matrix (Random Forest — the primary model)
# ---------------------------------------------------------
cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(5, 4.5))
import seaborn as sns
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Churned", "Churned"], yticklabels=["Not Churned", "Churned"])
plt.title("Random Forest — Confusion Matrix (Churn)")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/12_churn_confusion_matrix.png", dpi=120)
plt.close()

# ---------------------------------------------------------
# ROC curve comparison
# ---------------------------------------------------------
plt.figure(figsize=(6, 5.5))
for name, proba in [("Logistic Regression", lr_proba), ("Random Forest", rf_proba)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve — Churn Prediction Models")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/13_churn_roc_curve.png", dpi=120)
plt.close()

# Feature importance (Random Forest)
importances = pd.Series(rf_clf.feature_importances_, index=feature_cols).sort_values()
plt.figure(figsize=(7, 4.5))
importances.plot(kind="barh", color="indianred")
plt.title("Random Forest Feature Importance (Churn Prediction)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/14_churn_feature_importance.png", dpi=120)
plt.close()

joblib.dump(rf_clf, "/home/claude/retail_project/models/churn_random_forest.pkl")
joblib.dump(log_reg, "/home/claude/retail_project/models/churn_logistic_regression.pkl")
joblib.dump(scaler, "/home/claude/retail_project/models/churn_scaler.pkl")
print("\nChurn models saved to models/. Charts saved to outputs/charts/.")
