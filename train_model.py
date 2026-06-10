import pandas as pd
from sklearn.cluster import KMeans
import pickle
import os

# Load Dataset
df = pd.read_csv("data/Mall_Customers.csv")

# Features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Train KMeans
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

kmeans.fit(X)

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Save Model
with open("models/kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

print("KMeans Model Saved Successfully!")