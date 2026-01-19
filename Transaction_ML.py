import pandas as pd
import sklearn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras

df = pd.read_csv("Supplier_transaction_data.csv")
print("done")

features = [
    "sale_id",
    "order_id",
    "sale_date",
    "channel",
    "quantity",
    "unit_price",
    "net_revenue",
    "sku_id",
    "product_name",
    "category",
    "sub_category",
    "brand",
    "supplier_id",
    "supplier_name",
    "region"
]

X = df[features]


scaler = StandardScaler()
# will get errors here string columns might need to be encoded will fix later
X_scaled = scaler.fit_transform(X)

print("done")
