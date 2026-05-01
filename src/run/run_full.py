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
from src.models.context_model import apply_context_boost
from src.explainability.explain import generate_explanation


def main():
    # Load + preprocess
    movies = preprocess_movies(load_movies())
    ratings = preprocess_ratings(load_ratings())

    # Models
    content_model = ContentBasedRecommender()
    content_model.fit(movies)

    collab_model = CollaborativeRecommender()
    collab_model.fit(ratings, movies)

    hybrid = HybridRecommender(content_model, collab_model)

    # Inputs
    user_id = 1
    movie_name = "Toy Story (1995)"
    mood = "happy"
    time_of_day = "night"

    # Recommendation
    recs = hybrid.recommend(user_id, movie_name, ratings, top_n=10)

    # Context
    recs = apply_context_boost(recs, mood, time_of_day)

    # Merge genres for explanation
    recs = recs.merge(movies[["movieId", "genres"]], on="movieId", how="left")

    print(f"\nFINAL RECOMMENDATIONS\n")

    for _, row in recs.head(5).iterrows():
        explanation = generate_explanation(row["genres"], mood, time_of_day)

        print(f"Movie: {row['title']}")
        print(f"Score: {round(row['final_score'], 3)}")
        print(f"{explanation}")
        print("-" * 50)


if __name__ == "__main__":
    main()