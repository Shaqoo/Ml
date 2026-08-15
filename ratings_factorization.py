import pandas as pd
import numpy as np

df = pd.read_csv('ratings.csv')

matrix = pd.pivot_table(
    df,
    values='rating',
    index='user_id',
    columns='movie_id',
    fill_value=0
)

print(matrix.head())

# matrix = matrix.fillna(0)

R = matrix.values
print(R)

num_users, num_movies = matrix.shape
num_factors = 5
U = np.random.rand(num_users, num_factors)
V = np.random.rand(num_movies, num_factors)

print("U shape:", U.shape)
print("V shape:", V.shape)

predicted = U @ V.T
print("Predicted shape:", predicted.shape)

mask = R > 0
error = (R - predicted) * mask
print("Error shape:", error.shape)
mse = np.sum(error ** 2) / np.sum(mask)

print("MSE:", mse)

# print(num_users,num_movies)