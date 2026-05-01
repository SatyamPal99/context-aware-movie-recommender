import pandas as pd
from surprise import Dataset, Reader, SVD


class CollaborativeRecommender:
    def __init__(self):
        self.model = SVD(n_factors=50, n_epochs=10)  # faster config
        self.movies = None

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
        self.movies = movies.copy()

        #  IMPORTANT: Reduce dataset size
        ratings = ratings.sample(n=300000, random_state=42)

        # Optional: keep active users only
        user_counts = ratings["userId"].value_counts()
        active_users = user_counts[user_counts > 20].index
        ratings = ratings[ratings["userId"].isin(active_users)]

        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(
            ratings[["userId", "movieId", "rating"]],
            reader
        )

        trainset = data.build_full_trainset()
        self.model.fit(trainset)

    def recommend_for_user(self, user_id: int, ratings: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        rated_movies = set(ratings[ratings["userId"] == user_id]["movieId"])

        candidates = self.movies[~self.movies["movieId"].isin(rated_movies)].copy()

        # Limit predictions for speed
        candidates = candidates.sample(n=min(5000, len(candidates)))

        candidates["predicted_rating"] = candidates["movieId"].apply(
            lambda movie_id: self.model.predict(user_id, movie_id).est
        )

        return candidates.sort_values("predicted_rating", ascending=False).head(top_n)