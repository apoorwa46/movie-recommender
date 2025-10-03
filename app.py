import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from scipy.sparse import load_npz
from sklearn.neighbors import NearestNeighbors

# ------------------- TMDb API Config -------------------
TMDB_API_KEY = "5bed42cbda996262f994913ba8d9b65d"
TMDB_BASE_URL = "https://api.themoviedb.org/3/search/movie"

def get_movie_poster(title, year=None):
    """
    Fetch movie poster from TMDb API. 
    If year is provided, tries to match the release year for older movies.
    """
    params = {"api_key": TMDB_API_KEY, "query": title}
    if year:
        params["year"] = year
    try:
        response = requests.get(TMDB_BASE_URL, params=params, timeout=5)
        data = response.json()
        results = data.get('results', [])
        if results:
            # If year is provided, try to match the closest result
            if year:
                for movie in results:
                    release_date = movie.get('release_date', '')
                    if release_date.startswith(str(year)):
                        poster_path = movie.get('poster_path')
                        if poster_path:
                            return f"https://image.tmdb.org/t/p/w200{poster_path}"
            # Fallback to first result
            poster_path = results[0].get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w200{poster_path}"
    except:
        pass
    # Fallback placeholder
    return "https://via.placeholder.com/200x300?text=No+Image"

# ------------------- Load Models -------------------
matrix_sparse = load_npz("models/user_movie_matrix_sparse.npz")
with open("models/movie_id_map.pkl", "rb") as f:
    movie_id_map = pickle.load(f)
with open("models/movies.pkl", "rb") as f:
    movies = pickle.load(f)

# ------------------- Train Item-based KNN -------------------
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(matrix_sparse.T)  # transpose for item-based CF

# ------------------- Streamlit Config -------------------
st.set_page_config(page_title="Movie Recommendation 🎬", layout="wide")
st.title("🎬 Movie Recommendation System")
st.markdown("**Discover movies similar to your favorites with posters and genres!**")

# ------------------- Sidebar -------------------
st.sidebar.header("ℹ️ About")
st.sidebar.info("""
This app recommends movies using **item-based KNN collaborative filtering** on MovieLens 25M dataset.  
Select movies you liked, and get personalized recommendations with posters.  
""")

# ------------------- Tabs -------------------
tab1, tab2, tab3 = st.tabs(["🏠 Home", "🎯 Recommend", "📊 Insights"])

# ------------------- HOME TAB -------------------
with tab1:
    st.header("Welcome to Movie Recommendation System")
    st.markdown("""
    ### How it works
    1. Select movies you liked.  
    2. The system finds **similar movies based on other users' behavior**.  
    3. Get a list of movies you'll likely enjoy with their posters!  
    """)
    st.image("https://via.placeholder.com/800x400?text=Movie+Recommendation+System", use_container_width=True)

# ------------------- RECOMMEND TAB -------------------
with tab2:
    st.header("Select Movies You Liked")
    liked_movies = st.multiselect("Start typing movie names...", movies['title'].tolist())

    if st.button("Get Recommendations"):
        if len(liked_movies) == 0:
            st.warning("Please select at least one movie!")
        else:
            # Map selected titles to movie IDs
            selected_movie_ids = movies[movies['title'].isin(liked_movies)]['movieId'].tolist()

            # ------------------- Item-based KNN -------------------
            recommended_ids = set()
            movie_index_to_id = {v: k for k, v in movie_id_map.items()}

            for mid in selected_movie_ids:
                if mid in movie_id_map:
                    movie_vec_index = movie_id_map[mid]
                    distances, indices = model_knn.kneighbors(
                        matrix_sparse.T[movie_vec_index], n_neighbors=6
                    )
                    for i in indices.flatten():
                        rec_id = movie_index_to_id.get(i)
                        if rec_id and rec_id not in selected_movie_ids:
                            recommended_ids.add(rec_id)

            recommended_titles = movies[movies['movieId'].isin(recommended_ids)]['title'].tolist()[:10]

            # ------------------- Display recommendations -------------------
            st.subheader("🎬 Recommended Movies:")
            if recommended_titles:
                num_cols = min(len(recommended_titles), 5)
                cols = st.columns(num_cols)
                for idx, title in enumerate(recommended_titles[:num_cols]):
                    # Extract year from title (e.g., "Sabrina (1995)")
                    year = None
                    title_row = movies[movies['title'] == title]
                    if not title_row.empty:
                        year_str = title_row.iloc[0]['title'][-5:-1]
                        if year_str.isdigit():
                            year = int(year_str)
                    poster_url = get_movie_poster(title, year)
                    with cols[idx]:
                        st.image(poster_url, use_container_width=True)
                        st.markdown(f"**{title}**")
            else:
                st.info("No recommendations found. Try selecting more movies or different genres.")

            # ------------------- Genre distribution -------------------
            if 'genres' in movies.columns:
                rec_movies_df = movies[movies['title'].isin(recommended_titles)]
                genre_counts = {}
                for g in rec_movies_df['genres']:
                    for genre in g.split('|'):
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1

                fig, ax = plt.subplots(figsize=(8,4))
                ax.bar(genre_counts.keys(), genre_counts.values(), color='skyblue')
                ax.set_title("🎭 Genre Distribution of Recommendations")
                ax.set_ylabel("Count")
                plt.xticks(rotation=45)
                st.pyplot(fig)

# ------------------- INSIGHTS TAB -------------------
with tab3:
    st.header("🎯 Insights & Tips")
    st.markdown("""
    - Recommendations are based on **movies similar to the ones you like**.  
    - Using **item-based KNN**, the system finds movies that users who liked your movies also enjoyed.  
    - Posters are fetched dynamically from **TMDb API**, with fallback for older movies.  
    - Add more liked movies for more personalized recommendations.  
    - Explore different genres to diversify your watchlist.  
""")
