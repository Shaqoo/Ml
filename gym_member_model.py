import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


df = pd.read_csv('gym_members.csv')
# print(df.head())

X = df[["Workout_Frequency","Avg_Workout_Duration"]]

scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)

# inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled).inertia_ 
#     for k in range(1, 11)]

# plt.plot(range(1, 11), inertias, marker='o')
# plt.title('The Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('Inertia')
# plt.show()

# inertias = []
# silhouttes = []

# for k in range(2, 11):
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     labels = kmeans.fit_predict(X_scaled)
#     inertias.append(kmeans.inertia_)

#     silhouette_score_value = silhouette_score(X_scaled, labels)
#     silhouttes.append(silhouette_score_value)


# print(inertias)
# print(silhouttes)


# k_range = range(2, 11)

# plt.figure(figsize=(14, 5))

# # Plot 1: The Elbow Method (Inertia)
# plt.subplot(1, 2, 1)
# plt.plot(k_range, inertias, marker='o', color='b', linestyle='--')
# plt.xlabel('Number of Clusters (K)')
# plt.ylabel('Inertia')
# plt.title('Elbow Method for Optimal K')
# plt.axvline(x=4, color='r', linestyle=':', label='Elbow Point (K=4)')
# plt.legend()

# # Plot 2: Silhouette Scores
# plt.subplot(1, 2, 2)
# plt.plot(k_range, silhouttes, marker='s', color='g', linestyle='-')
# plt.xlabel('Number of Clusters (K)')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Scores for Optimal K')
# plt.axvline(x=4, color='r', linestyle=':', label='Peak Score (K=4)')
# plt.legend()

# plt.tight_layout()
# plt.show()




kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.title("Workout Frequency vs Average Workout Duration")
plt.xlabel("Workout Frequency")
plt.ylabel("Average Workout Duration")

centroids_original = scalar.inverse_transform(kmeans.cluster_centers_)
print(centroids_original)
print(kmeans.cluster_centers_)
# print(centroids_original)
plt.scatter(centroids_original[:, 0], centroids_original[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.legend()
plt.show()

# #🧠 Exercise: The Fitness Center Member Segmenter

# Imagine a fitness company has a dataset called gym_members.csv.

# It contains:

# MemberID
# Age
# Monthly_Income
# Workout_Frequency
# Avg_Workout_Duration

# The business wants to understand what types of gym members they have so they can create better membership plans.

# Your task

# Use K-Means clustering to segment the members based on:

# Workout_Frequency — workouts per week
# Avg_Workout_Duration — average minutes per workout

# Ignore MemberID, Age, and Monthly_Income.

