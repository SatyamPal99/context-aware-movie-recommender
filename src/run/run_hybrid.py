import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.load_data import load_movies, load_ratings
from src.data_processing.preprocess import preprocess_movies, preprocess_ratings
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender
from src.models.hybrid import HybridRecommender


def main():
    movies = preprocess_movies(load_movies())
    ratings = preprocess_ratings(load_ratings())

    content_model = ContentBasedRecommender()
    content_model.fit(movies)

    collab_model = CollaborativeRecommender()
    collab_model.fit(ratings, movies)

    hybrid = HybridRecommender(content_model, collab_model, alpha=0.5)

    user_id = 1
    movie_name = "Toy Story (1995)"

    print(f"\nHybrid Recommendations for User {user_id} based on '{movie_name}'")
    recs = hybrid.recommend(user_id, movie_name, ratings, top_n=5)

    print(recs[["movieId", "title", "final_score"]])


if __name__ == "__main__":
    main()