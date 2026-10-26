"""
WEEK 10: Product Recommendation Engine
Builds a collaborative filtering recommender using matrix factorization
(Truncated SVD) on the customer-product interaction matrix, and a
popularity-based fallback for the cold-start problem (new customers with
no purchase history).
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import joblib

df = pd.read_csv("/home/claude/retail_project/data/transactions.csv")
products = pd.read_csv("/home/claude/retail_project/data/products.csv")

# ---------------------------------------------------------
# 1. Build the customer-product interaction matrix (implicit feedback:
#    quantity purchased, since we don't have explicit ratings)
# ---------------------------------------------------------
interactions = df.groupby(["customer_id", "product_id"])["quantity"].sum().reset_index()

cust_ids = interactions["customer_id"].unique()
prod_ids = interactions["product_id"].unique()
cust_idx = {c: i for i, c in enumerate(cust_ids)}
prod_idx = {p: i for i, p in enumerate(prod_ids)}

from scipy.sparse import csr_matrix
rows = interactions["customer_id"].map(cust_idx)
cols = interactions["product_id"].map(prod_idx)
vals = interactions["quantity"].values
matrix = csr_matrix((vals, (rows, cols)), shape=(len(cust_ids), len(prod_ids)))

print(f"Customer-product interaction matrix: {matrix.shape[0]} customers x {matrix.shape[1]} products")
print(f"Sparsity: {100 * (1 - matrix.nnz / (matrix.shape[0]*matrix.shape[1])):.2f}% empty")

# ---------------------------------------------------------
# 2. Matrix factorization (Truncated SVD -> latent customer/product factors)
# ---------------------------------------------------------
N_FACTORS = 20
svd = TruncatedSVD(n_components=N_FACTORS, random_state=42)
customer_factors = svd.fit_transform(matrix)          # (n_customers, N_FACTORS)
product_factors = svd.components_.T                    # (n_products, N_FACTORS)

print(f"Explained variance captured by {N_FACTORS} latent factors: "
      f"{svd.explained_variance_ratio_.sum()*100:.1f}%")

# Reconstructed (predicted) interaction scores
predicted_scores = customer_factors @ product_factors.T  # (n_customers, n_products)

# ---------------------------------------------------------
# 3. Popularity fallback for cold-start (new customers with no history)
# ---------------------------------------------------------
popularity = interactions.groupby("product_id")["quantity"].sum().sort_values(ascending=False)
top_popular_products = popularity.head(10).index.tolist()

# ---------------------------------------------------------
# 4. Recommendation function
# ---------------------------------------------------------
def recommend_products(customer_id, n=5):
    """Return top-N recommended product IDs for a given customer."""
    if customer_id not in cust_idx:
        # Cold start: customer has no purchase history -> popularity fallback
        return top_popular_products[:n], "popularity_fallback"

    c_i = cust_idx[customer_id]
    scores = predicted_scores[c_i].copy()

    # Exclude products already purchased
    already_bought = matrix[c_i].nonzero()[1]
    scores[already_bought] = -np.inf

    top_indices = np.argsort(scores)[::-1][:n]
    idx_to_prod = {v: k for k, v in prod_idx.items()}
    recommended = [idx_to_prod[i] for i in top_indices]
    return recommended, "collaborative_filtering"

# ---------------------------------------------------------
# 5. Demonstrate on a few sample customers
# ---------------------------------------------------------
print("\nSAMPLE RECOMMENDATIONS")
print("=" * 60)
sample_customers = list(cust_ids[:3]) + ["CUST99999"]  # last one = cold start example
cat_map = dict(zip(products.product_id, products.category))

results_log = []
for cust in sample_customers:
    recs, method = recommend_products(cust, n=5)
    rec_categories = [cat_map.get(p, "?") for p in recs]
    print(f"\nCustomer: {cust}  (method: {method})")
    for p, c in zip(recs, rec_categories):
        print(f"   -> {p}  [{c}]")
    results_log.append({"customer_id": cust, "method": method, "recommendations": recs})

pd.DataFrame(results_log).to_csv(
    "/home/claude/retail_project/outputs/sample_recommendations.csv", index=False)

# ---------------------------------------------------------
# 6. Save the recommendation model artifacts
# ---------------------------------------------------------
joblib.dump({
    "svd": svd,
    "customer_factors": customer_factors,
    "product_factors": product_factors,
    "cust_idx": cust_idx,
    "prod_idx": prod_idx,
    "matrix": matrix,
    "top_popular_products": top_popular_products,
}, "/home/claude/retail_project/models/recommendation_engine.pkl")

print("\nRecommendation engine saved to models/recommendation_engine.pkl")
print("Sample recommendations saved to outputs/sample_recommendations.csv")
