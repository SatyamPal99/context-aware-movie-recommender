import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.load_data import load_movies, load_ratings
from src.data_processing.preprocess import preprocess_movies, preprocess_ratings
from src.models.collaborative import CollaborativeRecommender


def main():
    movies = preprocess_movies(load_movies())
    ratings = preprocess_ratings(load_ratings())

    model = CollaborativeRecommender()
    model.fit(ratings, movies)

    user_id = 1  # change if needed

    print(f"\nRecommendations for User: {user_id}")
    recs = model.recommend_for_user(user_id, ratings, top_n=5)
    print(recs[["movieId", "title", "predicted_rating"]])


if __name__ == "__main__":
    main()