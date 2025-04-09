import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import logging
from utils.preprocessing import load_data, preprocess_data
from models import model_kmeans

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load and preprocess data
logging.info("Loading and preprocessing data...")
df = load_data("data/mall_customers.csv")
X = preprocess_data(df, save_cleaned_path="data/cleaned_mall_customers.csv")

# Show Elbow Method plot
logging.info("Plotting elbow method to determine optimal k...")
model_kmeans.plot_elbow_method(X)

# Choose a value for k based on elbow plot (e.g., k=5)
chosen_k = 5
logging.info(f"Running KMeans clustering with k={chosen_k}...")
model, labels = model_kmeans.run_kmeans(X, n_clusters=chosen_k)


score = model_kmeans.calculate_silhouette_score(X, n_clusters=chosen_k)
print(f"Silhouette Score for k={chosen_k}: {score:.2f}")
