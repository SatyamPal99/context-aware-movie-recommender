import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.load_data import load_movies
from src.data_processing.preprocess import preprocess_movies
from src.models.content_based import ContentBasedRecommender


def main():
    movies = preprocess_movies(load_movies())

    model = ContentBasedRecommender()
    model.fit(movies)

    movie_name = "Toy Story (1995)"  # change if needed

    print(f"\nRecommendations for: {movie_name}")
    recs = model.recommend(movie_name, top_n=5)
    print(recs)


if __name__ == "__main__":
    main()