import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class ContentBasedRecommender:
    def __init__(self):
        self.movies = None
        self.tfidf = None
        self.tfidf_matrix = None
        self.title_to_index = None

    def fit(self, movies: pd.DataFrame) -> None:
        """
        Train the content-based model using title + genres
        """

        # Copy data
        self.movies = movies.reset_index(drop=True).copy()

        # Combine features (VERY IMPORTANT)
        self.movies["combined"] = (
            self.movies["title"].fillna("") + " " +
            self.movies["genres"].fillna("")
        )

        # TF-IDF vectorization
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies["combined"])

        # Mapping from title → index
        self.title_to_index = pd.Series(
            self.movies.index,
            index=self.movies["title"]
        ).drop_duplicates()

    def recommend(self, movie_title: str, top_n: int = 5) -> pd.DataFrame:
        """
        Recommend similar movies based on content
        """

        if movie_title not in self.title_to_index:
            raise ValueError(f"Movie '{movie_title}' not found.")

        # Get index of selected movie
        idx = self.title_to_index[movie_title]

        # Compute similarity ONLY for this movie (memory efficient)
        cosine_scores = linear_kernel(
            self.tfidf_matrix[idx:idx + 1],
            self.tfidf_matrix
        ).flatten()

        # Sort scores
        scores = list(enumerate(cosine_scores))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        # Skip itself (index 0)
        scores = scores[1: top_n + 1]

        movie_indices = [i[0] for i in scores]

        # Prepare result
        result = self.movies.iloc[movie_indices][
            ["movieId", "title", "genres"]
        ].copy()

        result["score"] = [i[1] for i in scores]

        return result