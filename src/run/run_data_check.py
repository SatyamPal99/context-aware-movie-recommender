import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.load_data import load_movies, load_ratings
from src.data_processing.preprocess import preprocess_movies, preprocess_ratings


def main():
    movies = load_movies()
    ratings = load_ratings()

    print("Raw Data:")
    print("Movies:", movies.shape)
    print("Ratings:", ratings.shape)

    movies = preprocess_movies(movies)
    ratings = preprocess_ratings(ratings)

    print("\nAfter Preprocessing:")
    print("Movies:", movies.shape)
    print("Ratings:", ratings.shape)

    print("\nSample Movies:")
    print(movies.head())

    print("\nSample Ratings:")
    print(ratings.head())


if __name__ == "__main__":
    main()