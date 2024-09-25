import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Load and preprocess data
def load_and_preprocess_data(file_path):
    d = pd.read_csv(file_path)
    d = d.dropna()
    return d

# Select features and scale data
def prepare_features(df, features):
    data = df[features]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, columns=features)

# Find optimal number of clusters using elbow method and silhouette score
def find_optimal_clusters(data, max_clusters=10):
    inertias = []
    silhouette_scores = []
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))
    
    return inertias, silhouette_scores

# Perform K-means clustering
def perform_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)
    return cluster_labels, kmeans.cluster_centers_

# Create 3D scatter plot
def create_3d_scatter(df, features, cluster_column):
    fig = px.scatter_3d(df, x=features[0], y=features[1], z=features[2],
                        color=cluster_column, hover_data=features)
    fig.update_layout(scene=dict(xaxis_title=features[0],
                                 yaxis_title=features[1],
                                 zaxis_title=features[2]))
    return fig

# Create elbow plot
def create_elbow_plot(inertias, silhouette_scores):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Elbow Method", "Silhouette Score"))
    
    fig.add_trace(go.Scatter(x=list(range(2, len(inertias) + 2)), y=inertias,
                             mode='lines+markers', name='Inertia'), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(2, len(silhouette_scores) + 2)), y=silhouette_scores,
                             mode='lines+markers', name='Silhouette Score'), row=1, col=2)
    
    fig.update_layout(height=500, width=1000, title_text="Optimal number of clusters")
    fig.update_xaxes(title_text="Number of clusters")
    fig.update_yaxes(title_text="Inertia", row=1, col=1)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
    
    return fig

# Perform PCA and visualize results
def perform_pca_and_visualize(data, cluster_labels):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = cluster_labels
    
    fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster')
    fig.update_layout(title="PCA visualization of clusters")
    
    return fig, pca.explained_variance_ratio_

# Main function
def main():
    # Load and preprocess data
    df = load_and_preprocess_data("E:/jyotindra/Ai-ML/credit-card-clustering/data.csv")
    
    # Select features for clustering
    features = ["BALANCE", "PURCHASES", "CREDIT_LIMIT"]
    scaled_data = prepare_features(df, features)
    
    # Find optimal number of clusters
    inertias, silhouette_scores = find_optimal_clusters(scaled_data)
    elbow_plot = create_elbow_plot(inertias, silhouette_scores)
    elbow_plot.show()
    
    # Choose optimal number of clusters (you can adjust this based on the elbow plot)
    n_clusters = 5
    
    # Perform clustering
    cluster_labels, cluster_centers = perform_clustering(scaled_data, n_clusters)
    
    # Add cluster labels to the original dataframe
    df['Cluster'] = cluster_labels
    
    # Create 3D scatter plot
    scatter_3d = create_3d_scatter(df, features, 'Cluster')
    scatter_3d.show()
    
    # Perform PCA and visualize results
    pca_plot, explained_variance = perform_pca_and_visualize(scaled_data, cluster_labels)
    pca_plot.show()
    print(f"Explained variance ratio: {explained_variance}")
    
    # Analyze cluster characteristics
    cluster_summary = df.groupby('Cluster')[features].mean()
    print("\nCluster Summary:")
    print(cluster_summary)

if __name__ == "__main__":
    main()