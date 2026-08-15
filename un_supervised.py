from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
import numpy as np
from matplotlib.image import imread
from sklearn.ensemble import IsolationForest
import os
from typing import Any

# X = np.array([[1, 2], [1, 4], [1, 0],
#                             [10, 2], [10, 4], [10, 0],[1, 0], [1, 4],   
#     [10, 1], [10, 3]  ])

# kmeans = KMeans(n_clusters=2, random_state=42)

# kmeans.fit(X)

# labels = kmeans.predict([[0, 0], [12, 3]])

# print(labels)

# print(kmeans.cluster_centers_)

# print(kmeans.inertia_)

# print(kmeans.predict([[5, 2]]))


# K_range = range(1, 11)
# inertias = []

# for k in K_range:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(X)
#     inertias.append(kmeans.inertia_)

# plt.figure(figsize=(8, 4))
# plt.plot(K_range, inertias, marker='o', linestyle='-', color='b', linewidth=2)
# plt.title('The Elbow Method for Your Custom Dataset', fontsize=12)
# plt.xlabel('Number of Clusters (K)', fontsize=10)
# plt.ylabel('Inertia (Total WCSS)', fontsize=10)
# plt.xticks(K_range)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()


# K_range = range(1, 11)
# inertias = []
# for k in K_range:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
#     kmeans.fit(X)
#     inertias.append(kmeans.inertia_)

# # 3. Fit final optimal K=2 model for the scatter plot
# kmeans_final = KMeans(n_clusters=2, random_state=42, n_init='auto')
# kmeans_final.fit(X)
# labels = kmeans_final.labels_
# centroids = kmeans_final.cluster_centers_

# # 4. Create 1 Figure with 2 Axes (Subplots side-by-side)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# # --- AXIS 1: The Elbow Method ---
# ax1.plot(K_range, inertias, marker='o', linestyle='-', color='b', linewidth=2)
# ax1.set_title('The Elbow Method for Your Custom Dataset', fontsize=12)
# ax1.set_xlabel('Number of Clusters (K)', fontsize=10)
# ax1.set_ylabel('Inertia (Total WCSS)', fontsize=10)
# ax1.set_xticks(K_range)
# ax1.grid(True, linestyle='--', alpha=0.6)

# # --- AXIS 2: The Clustered Data ---
# # Plot points colored by cluster labels
# scatter = ax2.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=100, edgecolors='k', zorder=2)
# # Plot red X centroids
# ax2.scatter(centroids[:, 0], centroids[:, 1], marker='X', c='red', s=200, label='Centroids', edgecolors='white', zorder=3)

# ax2.set_title('Visualizing the Final Clusters (K = 2)', fontsize=12)
# ax2.set_xlabel('X Coordinate', fontsize=10)
# ax2.set_ylabel('Y Coordinate', fontsize=10)
# ax2.set_xlim(-1, 12)
# ax2.set_ylim(-1, 5)
# ax2.grid(True, linestyle='--', alpha=0.5, zorder=1)
# ax2.legend(loc='upper right')

# # Adjust layout so plots don't overlap and show
# plt.tight_layout()
# plt.show()

# X_3D = np.random.rand(100, 3)  

# pca = PCA(n_components=2)

# X_2D = pca.fit_transform(X_3D)

 
# print(f"Original Shape: {X_3D.shape}")    
# print(f"Compressed Shape: {X_2D.shape}")  
# print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

# digits_dt: Any = load_digits()
# X_digits = digits_dt.data
# y_digits = digits_dt.target

# print(f"Original Shape: {X_digits.shape}")
# pca = PCA(n_components=2)
# X_2D = pca.fit_transform(X_digits)
# print(f"Compressed Shape: {X_2D.shape}")

# print(f"Explained Variance Ratio: {sum(pca.explained_variance_ratio_)}")

# plt.scatter(X_2D[:, 0], X_2D[:, 1], c=y_digits, cmap='tab10', alpha=0.6)
# plt.colorbar()
# plt.show()

# merge_history = linkage(X, method='ward')

# dendrogram(merge_history)

# plt.xlabel('Data Points')
# plt.ylabel('Distance')
# plt.title('Dendrogram')
# plt.show()

# X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
# merge_history = linkage(X, method='ward')
 
# dendrogram(merge_history)
# plt.show()


# image = imread("sample.jpg")
# print(image.shape)
# h, w , c = image.shape

# pixels = image.reshape(-1, 3)
# print(pixels.shape)

# kmeans = KMeans(n_clusters=4, random_state=42)
# kmeans.fit(pixels)
# # print(kmeans.cluster_centers_)
# # print(kmeans.inertia_)
# # print(kmeans.labels_)


# new_pixels = kmeans.cluster_centers_[kmeans.labels_]
# quantized_image = new_pixels.reshape(h, w, c).astype(np.uint8)

# plt.imshow(quantized_image)
# plt.axis('off')
# plt.show()
# clusters = [2, 5, 10]
# image = imread("sample2.jpg")
# h, w, c = image.shape

# plt.figure(figsize=(12, 10), layout="constrained")  

# row = 2
# col = 2
# position = 1


# plt.subplot(row, col, position)
# plt.imshow(image)
# plt.title('Original Image')
# plt.axis('off')  

# pixels = image.reshape(-1, 3)

# for k in clusters:
#     position += 1
    
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(pixels)
#     score = silhouette_score(pixels, kmeans.labels_, sample_size=10000, random_state=42)
#     print(f"Silhouette Score for k = {k}: {score}")

#     new_pixels = kmeans.cluster_centers_[kmeans.labels_]
#     quantized_image = new_pixels.reshape(h, w, c).astype(np.uint8)
    
#     plt.subplot(row, col, position)
#     plt.imshow(quantized_image)
#     plt.title(f'Quantized Image (K = {k})')
#     plt.axis('off')

# plt.show()

# 

# X = np.array([
#     [1.0, 1.0],
#     [1.1, 1.0],
#     [0.9, 1.1],

#     [8.0, 8.0],
#     [8.1, 8.2],
#     [7.9, 8.0],

#     [20.0, 20.0]
# ])

# model = DBSCAN(eps=0.3, min_samples=2)

# labels = model.fit_predict(X)

# print(labels)