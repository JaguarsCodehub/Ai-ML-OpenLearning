import pandas as pd
import numpy as np
from sklearn import cluster
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go

# Load and preprocess data
d = pd.read_csv("data.csv")
d = d.dropna()

# Select features for clustering
data = d[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]

# Apply MinMaxScaler to each column
for i in data.columns:
    MinMaxScaler(i)

# Perform K-means clustering
kmeans = KMeans(n_clusters=5)
cluster = kmeans.fit_predict(data)

# Add cluster labels to the original dataframe
a = d.copy()
a["CREDIT_CARD_SEGMENTS"] = cluster

# Map cluster numbers to descriptive labels
a["CREDIT_CARD_SEGMENTS"] = a["CREDIT_CARD_SEGMENTS"].map({
    0: "Cluster 1",
    1: "Cluster 2",
    2: "Cluster 3",
    3: "Cluster 4",
    4: "Cluster 5"
})

# Create 3D scatter plot
PLOT = go.Figure()
for i in list(a["CREDIT_CARD_SEGMENTS"].unique()):
    PLOT.add_trace(go.Scatter3d(
        x=a[a["CREDIT_CARD_SEGMENTS"] == i]["BALANCE"],
        y=a[a["CREDIT_CARD_SEGMENTS"] == i]["PURCHASES"],
        z=a[a["CREDIT_CARD_SEGMENTS"] == i]["CREDIT_LIMIT"],
        mode="markers",
        marker_size=6,
        marker_line_width=1,
        name=str(i)
    ))

PLOT.update_traces(hovertemplate='BALANCE: %{x} <br>PURCHASES: %{y} <br>CREDIT_LIMIT: %{z}')

PLOT.update_layout(
    width=2000,
    height=1000,
    autosize=True,
    showlegend=True,
    scene=dict(
        xaxis=dict(title='BALANCE', titlefont_color="black"),
        yaxis=dict(title='PURCHASES', titlefont_color="black"),
        zaxis=dict(title='CREDIT_LIMIT', titlefont_color="black")
    ),
    font=dict(family="Gilroy", color="black", size=12)
)

# Note: The plot is not displayed here, as it would typically be shown in a Jupyter notebook environment