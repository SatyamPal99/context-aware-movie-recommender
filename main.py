import pandas as pd

from src.models.new_user_recommender import NewUserRecommender
from src.models.existing_user_recommender import ExistingUserRecommender

from src.utils.fuzzy_search import find_closest_movie
from src.utils.time_utils import get_time_of_day

from src.realtime.update_model import add_clicked_movie, get_recent_history
from src.explainability.explain import generate_explanation


def main():
    print("\n========== MOVIE RECOMMENDATION SYSTEM ==========\n")

    # Load movie features
    movie_features = pd.read_csv("data/processed/movie_features.csv")

    # Models
    new_user_model = NewUserRecommender(movie_features)
    existing_user_model = ExistingUserRecommender(movie_features)

    # -------------------------
    # USER TYPE
    # -------------------------
    user_type = input("Are you a new user? (yes/no): ").strip().lower()
    mood = input("Enter mood (happy / serious / excited): ") or "happy"
    time_of_day = get_time_of_day()

    print(f"Detected Time of Day: {time_of_day}")

    # -------------------------
    # NEW USER FLOW
    # -------------------------
    if user_type == "yes":
        print("\n--- NEW USER SETUP ---")

        genres_input = input("Enter preferred genres (comma separated): ")
        preferred_genres = [g.strip() for g in genres_input.split(",")]

        liked_input = input("Enter 2-3 movies you like (comma separated): ")
        liked_movies_raw = [m.strip() for m in liked_input.split(",")]

        liked_movies = [
            find_closest_movie(m, movie_features["title"].tolist())
            for m in liked_movies_raw
        ]

        recommendations = new_user_model.recommend(
            preferred_genres=preferred_genres,
            liked_movies=liked_movies,
            mood=mood,
            top_n=10
        )

        user_id = input("Create a user ID for future use: ")

    # -------------------------
    # EXISTING USER FLOW
    # -------------------------
    else:
        print("\n--- EXISTING USER ---")

        user_id = input("Enter your user ID: ")

        recent_movies = get_recent_history(user_id)

        print(f"Your recent history: {recent_movies}")

        if not recent_movies:
            recent_input = input("Enter last 3-5 movies you watched: ")
            recent_movies_raw = [m.strip() for m in recent_input.split(",")]

            recent_movies = [
                find_closest_movie(m, movie_features["title"].tolist())
                for m in recent_movies_raw
            ]

        recommendations = existing_user_model.recommend(
            recent_movies=recent_movies,
            mood=mood,
            top_n=10
        )

    # -------------------------
    # OUTPUT
    # -------------------------
    print("\n🎬 TOP RECOMMENDATIONS:\n")

    for i, row in enumerate(recommendations.head(5).itertuples(), 1):
        explanation = generate_explanation(
            movie_genres=row.genres,
            mood=mood,
            recent_movies=get_recent_history(user_id),
            recommended_title=row.title,
            movie_features_df=movie_features
        )

        print(f"{i}. {row.title}")
        print(f"   Score: {round(row.final_score, 3)}")
        print(f"   {explanation}")
        print("-" * 60)

    # -------------------------
    # CLICK TRACKING
    # -------------------------
    choice = input("\nSelect a movie to watch (1-5): ")

    try:
        choice = int(choice)
        selected_movie = recommendations.iloc[choice - 1]["title"]

        print(f"\nYou selected: {selected_movie}")

        add_clicked_movie(user_id, selected_movie)

        print("✅ Your history has been updated!")

    except:
        print("Invalid selection")

    print("\n========== END ==========\n")


if __name__ == "__main__":
    main()