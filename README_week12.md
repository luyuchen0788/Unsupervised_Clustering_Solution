# Week 12 - Unsupervised Clustering (Customer Segmentation)

This project uses **KMeans clustering** to group mall customers based on their spending habits and income.  
The goal is to segment customers into meaningful clusters without using labeled data.

It includes:
- Data cleaning and standardization
- Elbow method and silhouette score evaluation
- Streamlit app for interactive clustering with real-time visualizations

---

##  Project Structure

- `data/mall_customers.csv` – raw input data
- `data/mall_customers_cleaned.csv` – cleaned and scaled dataset
- `utils/preprocessing.py` – handles data loading and preprocessing
- `models/model_kmeans.py` – KMeans training, elbow method, and silhouette score
- `main.py` – console version for clustering and evaluation
- `app.py` – Streamlit app for interactive segmentation
- `README.md` – project description and usage instructions

---

##  How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Launch the Streamlit app:

```bash
streamlit run app.py
```

3. Or run the script from terminal:

```bash
python main.py
```

---

## Clustering Summary

- Optimal number of clusters (k): **5**
- Silhouette Score: **0.29**
- Elbow method was used to help identify optimal cluster count

![Elbow Method](images/elbow_plot.png)

---

## Logging

This project uses Python's `logging` module to track the pipeline steps:  
loading data, preprocessing, elbow method plotting, and clustering evaluation.

---

## 👩‍🎓 Author

- Name: Luyu Chen  
- Student Number: 040986748