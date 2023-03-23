import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("K-Means Clustering")

st.sidebar.header("User Input Parameters")

# Read data from csv file
@st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_csv("Irisdata.csv.csv")
    return data

data = load_data()
# Show data on the app
st.subheader("Data")
st.write(data)

# Define features and target variable
features = data.iloc[:, :-1].values

# User-defined parameters
k = st.sidebar.slider("Number of clusters", 2, 10)
max_iter = st.sidebar.slider("Maximum number of iterations", 100, 1000, step=100)
n_init = st.sidebar.slider("Number of times k-means will be run with different centroid seeds", 1, 10)

# Build K-Means model
model = KMeans(n_clusters=k, max_iter=max_iter, n_init=n_init)
model.fit(features)

# Predict clusters
labels = model.predict(features)
# Plot clusters
st.subheader("Clustering Results")
fig, ax = plt.subplots()
ax.scatter(features[labels == 0, 0], features[labels == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
ax.scatter(features[labels == 1, 0], features[labels == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
ax.scatter(features[labels == 2, 0], features[labels == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
ax.scatter(features[labels == 3, 0], features[labels == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
ax.scatter(features[labels == 4, 0], features[labels == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
ax.set_title('Clusters')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend()
st.pyplot(fig)