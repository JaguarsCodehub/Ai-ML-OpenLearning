## Learning AI - ML From scratch

# AI-ML Roadmap

This advanced version of the code includes several improvements and additional analyses:
Modular structure: The code is organized into functions, making it more readable and maintainable.
Standardization: Instead of MinMaxScaler, we use StandardScaler, which is often preferred for K-means clustering.
Optimal number of clusters: We implement the elbow method and silhouette score to help determine the optimal number of clusters.
Improved visualization: We use Plotly Express for easier and more interactive plotting.
PCA analysis: We perform Principal Component Analysis (PCA) to visualize the clusters in 2D and understand the variance explained by the principal components.
Cluster characteristics: We analyze and print the mean values of features for each cluster.
Error handling and flexibility: The code is more robust and can handle different datasets and feature selections.

To use this code, you would run the script, and it will:
Load and preprocess the data
2. Show the elbow plot to help determine the optimal number of clusters
Perform K-means clustering
Display a 3D scatter plot of the clusters
Show a 2D PCA visualization of the clusters
6. Print the explained variance ratio from PCA
Display a summary of cluster characteristics
This advanced version provides a more comprehensive analysis of the credit card customer segmentation, offering multiple perspectives on the clustering results and helping to validate the choice of the number of clusters.