
import streamlit as st
import pandas as pd
import numpy as np
import requests

# Load data
@st.cache_data
def load_data():
    import os
    base_path = os.path.dirname(os.path.abspath(__file__))
    movies = pd.read_csv(os.path.join(base_path, 'film_fellow_movies.csv'))
    cosine_sim = np.load(os.path.join(base_path, 'cosine_final.npz'))['data'].astype(np.float32)
    return movies, cosine_sim

movies, cosine_sim = load_data()
TMDB_API_KEY = st.secrets["TMDB_API_KEY"]

# Recommendation function
def recommend_movies(title):
    matches = movies[movies['title_x'] == title]
    if len(matches) == 0:
        return pd.DataFrame()
    idx = matches.index[0]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:6]]  # Top 5
    return movies.iloc[movie_indices][['title_x', 'movie_id']]

# Streamlit UI
st.title("🎬 Film Fellow")
movie = st.selectbox("Search a movie", movies['title_x'])
if st.button("Recommend"):
    recs = recommend_movies(movie)
    cols = st.columns(5)
    for i, row in recs.iterrows():
        with cols[i % 5]:
            poster_url = f"https://image.tmdb.org/t/p/w500/{requests.get(f'https://api.themoviedb.org/3/movie/{row['movie_id']}?api_key={TMDB_API_KEY}').json()['poster_path']}"
            st.image(poster_url, width=150)
            st.caption(row['title_x'])
