import pandas as pd
import numpy as np
import sklearn
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from joblib import memory
import warnings


df = pd.read_csv("Supplier_transaction_data.csv")
df.columns = df.columns.str.strip()

df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
df["sale_year"] = df["sale_date"].dt.year
df["sale_month"] = df["sale_date"].dt.month
df["sale_day"] = df["sale_date"].dt.day

# Encode ONLY true categorical attributes (not IDs/names)
df_encoded = pd.get_dummies(
    df,
    columns=["channel", "category", "sub_category", "brand",
             "region"],
    drop_first=True
)

# Numeric-only for clustering
X = df_encoded.select_dtypes(include=["number"])

# Remove ID columns
X = X.drop(columns=["sale_id", "supplier_id"], errors="ignore")

# Clean inf/NaN
X = X.replace([np.inf, -np.inf], np.nan).dropna()

# Remove constant columns
X = X.loc[:, X.std() != 0]

# Scale
X_scaled = RobustScaler().fit_transform(X)

X_scaled = np.asarray(X_scaled)

# this will error due to the amount of matricies we have but silhouette scores are still produced
# scores = []
# for k in range(2, 11):
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
#     labels = kmeans.fit_predict(X_scaled)
#     score = silhouette_score(
#         X_scaled, labels, sample_size=5000, random_state=42)
#     scores.append(score)
#     print(f"k={k}, silhouette={score:.3f}")
# print("X_scaled shape:", X_scaled.shape)

# k=2 has the highest silhouette score but is limited to provided meaningful results we will be using k=4
