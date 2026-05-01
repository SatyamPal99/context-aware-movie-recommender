import pandas as pd


def preprocess_movies(movies: pd.DataFrame) -> pd.DataFrame:
    movies = movies.copy()
    movies["genres"] = movies["genres"].fillna("")
    movies["title"] = movies["title"].fillna("")
    return movies


def preprocess_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    ratings = ratings.copy()
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"])
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)
    return ratings