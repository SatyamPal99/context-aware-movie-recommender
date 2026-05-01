import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class HybridRecommender:
    def __init__(self, content_model, collaborative_model, alpha: float = 0.5):
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.alpha = alpha

    def recommend(self, user_id: int, seed_movie: str, ratings: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        content_df = self.content_model.recommend(seed_movie, top_n=50).copy()
        collab_df = self.collaborative_model.recommend_for_user(user_id, ratings, top_n=50).copy()

        content_df = content_df.rename(columns={"score": "content_score"})
        collab_df = collab_df.rename(columns={"predicted_rating": "collab_score"})

        merged = pd.merge(content_df, collab_df[["movieId", "collab_score"]], on="movieId", how="outer")

        merged["content_score"] = merged["content_score"].fillna(0)
        merged["collab_score"] = merged["collab_score"].fillna(0)

        # ✅ NORMALIZATION (IMPORTANT)
        scaler = MinMaxScaler()
        merged[["content_score", "collab_score"]] = scaler.fit_transform(
            merged[["content_score", "collab_score"]]
        )

        # Final score
        merged["final_score"] = self.alpha * merged["content_score"] + (1 - self.alpha) * merged["collab_score"]

        return merged.sort_values("final_score", ascending=False).head(top_n)