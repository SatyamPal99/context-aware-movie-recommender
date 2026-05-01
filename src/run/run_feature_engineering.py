import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.load_data import load_movies, load_ratings
from src.data_processing.preprocess import preprocess_movies, preprocess_ratings
from src.data_processing.feature_engineering import build_movie_features, save_movie_features


def main():
    print("Loading raw data...")
    movies = load_movies()
    ratings = load_ratings()

    print("Preprocessing...")
    movies = preprocess_movies(movies)
    ratings = preprocess_ratings(ratings)
    ratings = ratings.sample(n=1000000, random_state=42)

    print("Building movie feature table...")
    movie_features = build_movie_features(movies, ratings)

    print("\nMovie feature table created.")
    print(movie_features[["movieId", "title", "genres", "avg_rating", "rating_count", "popularity_score"]].head())

    save_movie_features(movie_features)
    print("\nSaved to: data/processed/movie_features.csv")


if __name__ == "__main__":
    main()