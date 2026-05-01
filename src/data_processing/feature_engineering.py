import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "movie_features.csv"


def build_movie_features(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Build a movie feature table using movie metadata + rating statistics.
    """

    movies = movies.copy()
    ratings = ratings.copy()

    movies["title"] = movies["title"].fillna("")
    movies["genres"] = movies["genres"].fillna("")

    rating_stats = (
        ratings.groupby("movieId")["rating"]
        .agg(avg_rating="mean", rating_count="count")
        .reset_index()
    )

    movie_features = movies.merge(rating_stats, on="movieId", how="left")

    movie_features["avg_rating"] = movie_features["avg_rating"].fillna(0)
    movie_features["rating_count"] = movie_features["rating_count"].fillna(0)

    # weighted popularity: average rating × log(1 + number of ratings)
    movie_features["popularity_score"] = (
        movie_features["avg_rating"] * np.log1p(movie_features["rating_count"])
    )

    return movie_features


def save_movie_features(
    movie_features: pd.DataFrame,
    output_path: str | Path = DEFAULT_FEATURES_PATH,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    movie_features.to_csv(output_path, index=False)