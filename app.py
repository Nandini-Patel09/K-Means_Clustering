import streamlit as st
import pickle
import numpy as np

# Load Model
with open("models/kmeans_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Customer Segmentation using K-Means")

st.write(
    """
    Enter Customer Details
    """
)

income = st.number_input(
    "Annual Income (k$)",
    min_value=0.0
)

score = st.number_input(
    "Spending Score (1-100)",
    min_value=0.0,
    max_value=100.0
)

if st.button("Predict Cluster"):

    data = np.array([[income, score]])

    cluster = model.predict(data)[0]

    st.success(f"Customer belongs to Cluster {cluster}")