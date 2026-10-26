"""
WEEK 6: Customer Segmentation (Unsupervised Learning)
Uses K-Means clustering on RFM + behavioral features to segment customers.
Determines optimal K via the elbow method, validates with silhouette score,
and profiles each resulting segment for business interpretation.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

CHART_DIR = "/home/claude/retail_project/outputs/charts"
rfm = pd.read_csv("/home/claude/retail_project/data/features_customer_rfm.csv")

cluster_features = ["recency_days", "frequency", "monetary", "avg_order_value",
                     "tenure_days", "distinct_categories"]
X = rfm[cluster_features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------
# 1. Elbow method to choose K
# ---------------------------------------------------------
inertias, sil_scores = [], []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.plot(list(K_range), inertias, marker="o", color="steelblue")
ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("Inertia (Elbow)", color="steelblue")
ax2 = ax1.twinx()
ax2.plot(list(K_range), sil_scores, marker="s", color="crimson")
ax2.set_ylabel("Silhouette Score", color="crimson")
plt.title("K-Means: Elbow Method & Silhouette Score")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/09_kmeans_elbow_silhouette.png", dpi=120)
plt.close()

print(f"Silhouette scores by K: {dict(zip(K_range, np.round(sil_scores, 3)))}")
# K=2 gives the mathematically highest silhouette score (it mainly separates
# active vs. churned customers), but collapses all active customers into one
# group -- not useful for targeted marketing. K=4 gives a small silhouette
# trade-off in exchange for business-actionable segments (a standard
# real-world judgment call: statistical optimum vs. business usefulness).
best_k = 4
print(f"Selected K (business-driven, balancing silhouette score with actionable "
      f"segment granularity): {best_k}")

# ---------------------------------------------------------
# 2. Final K-Means model
# ---------------------------------------------------------
final_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
rfm["segment"] = final_km.fit_predict(X_scaled)
final_sil = silhouette_score(X_scaled, rfm["segment"])
print(f"Final silhouette score with K={best_k}: {final_sil:.3f}")

# ---------------------------------------------------------
# 3. Segment profiling
# ---------------------------------------------------------
profile = rfm.groupby("segment")[cluster_features + ["churned"]].mean().round(1)
profile["num_customers"] = rfm.groupby("segment").size()
print("\nSEGMENT PROFILES")
print(profile.to_string())

# Business-friendly labels based on RFM characteristics (rank by monetary)
order = profile.sort_values("monetary", ascending=False).index.tolist()
label_map = {}
labels_pool = ["High-Value Loyal", "Steady Regular", "Occasional Buyer", "At-Risk / Low-Engagement"]
for i, seg in enumerate(order):
    label_map[seg] = labels_pool[i] if i < len(labels_pool) else f"Segment {seg}"
rfm["segment_label"] = rfm["segment"].map(label_map)
profile["segment_label"] = profile.index.map(label_map)

profile.to_csv("/home/claude/retail_project/outputs/segment_profiles.csv")
rfm.to_csv("/home/claude/retail_project/data/features_customer_rfm_segmented.csv", index=False)

# ---------------------------------------------------------
# 4. Visualize segments (monetary vs frequency, colored by segment)
# ---------------------------------------------------------
plt.figure(figsize=(8, 5.5))
sns.scatterplot(data=rfm, x="frequency", y="monetary", hue="segment_label", palette="Set2", s=45)
plt.title(f"Customer Segments (K-Means, K={best_k})")
plt.xlabel("Purchase Frequency"); plt.ylabel("Monetary Value")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/10_customer_segments_scatter.png", dpi=120)
plt.close()

# Segment size bar chart
plt.figure(figsize=(7, 4.5))
seg_counts = rfm["segment_label"].value_counts()
sns.barplot(x=seg_counts.values, y=seg_counts.index, color="mediumseagreen")
plt.title("Customer Count per Segment")
plt.xlabel("Number of Customers")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/11_segment_sizes.png", dpi=120)
plt.close()

joblib.dump(final_km, "/home/claude/retail_project/models/kmeans_segmentation.pkl")
joblib.dump(scaler, "/home/claude/retail_project/models/segmentation_scaler.pkl")
print("\nSegmentation model and scaler saved to models/. Charts saved to outputs/charts/.")
