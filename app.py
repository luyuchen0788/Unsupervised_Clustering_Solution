import streamlit as st
import pandas as pd
from utils.preprocessing import load_data, preprocess_data
from models import model_kmeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set Streamlit title
st.title(" Mall Customer Segmentation App")
st.write("This app uses **KMeans Clustering** to segment mall customers.")

# Load and preprocess data
logging.info("Loading and preprocessing data...")
df = load_data("data/mall_customers.csv")
X = preprocess_data(df)

# User input: number of clusters
k = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=5)

# Run clustering
logging.info(f"Running KMeans with k={k}")
model, labels = model_kmeans.run_kmeans(X, n_clusters=k)

# Optional: Silhouette Score
score = model_kmeans.calculate_silhouette_score(X, n_clusters=k)
st.write(f"**Silhouette Score** for k={k}: {score:.2f}")

# Apply PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot clusters
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", s=50)
ax.set_title("Customer Segments (PCA 2D View)")
st.pyplot(fig)
