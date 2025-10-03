import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, save_npz
import pickle
import os

os.makedirs("models", exist_ok=True)

# ------------------- Load Data -------------------
ratings = pd.read_csv('data/ratings.csv')   # userId, movieId, rating, timestamp
movies = pd.read_csv('data/movies.csv')     # movieId, title, genres

# ------------------- Keep only popular movies -------------------
popular_movies = ratings.groupby('movieId').size()
popular_movies = popular_movies[popular_movies >= 1000].index  # keep movies with >=1000 ratings
ratings = ratings[ratings['movieId'].isin(popular_movies)]

# ------------------- Create Sparse User-Movie Matrix -------------------
user_ids = ratings['userId'].unique()
movie_ids = ratings['movieId'].unique()
user_id_map = {uid:i for i, uid in enumerate(user_ids)}
movie_id_map = {mid:i for i, mid in enumerate(movie_ids)}

rows = ratings['userId'].map(user_id_map)
cols = ratings['movieId'].map(movie_id_map)
data = ratings['rating']

matrix_sparse = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(movie_ids)))

# ------------------- Train KNN -------------------
# Train KNN on movie vectors (transpose)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(matrix_sparse.T)  # transpose for item-based


# ------------------- Save Models -------------------
save_npz("models/user_movie_matrix_sparse.npz", matrix_sparse)
with open("models/user_id_map.pkl", "wb") as f:
    pickle.dump(user_id_map, f)
with open("models/movie_id_map.pkl", "wb") as f:
    pickle.dump(movie_id_map, f)
with open("models/movies.pkl", "wb") as f:
    pickle.dump(movies, f)
with open("models/knn_model.pkl", "wb") as f:
    pickle.dump(model_knn, f)

print("✅ KNN model trained on MovieLens 25M popular movies (sparse) and saved successfully!")
