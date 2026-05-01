import pandas as pd
import random
import time
import matplotlib.pyplot as plt

from src.models.existing_user_recommender import ExistingUserRecommender
from src.evaluation.metrics import genre_coverage, ndcg_at_k_from_scores


# -------------------------
# CONFIG
# -------------------------
NUM_USERS = 50              # more users = better evaluation
TOP_K = 10
MIN_HISTORY = 6
SAMPLE_SIZE = 25000         # larger subset (balanced approach)
RANDOM_SEED = 42


# -------------------------
# LOAD FULL DATASET
# -------------------------
movie_features_full = pd.read_csv("data/processed/movie_features.csv")

print(f"Total movies in dataset: {len(movie_features_full)}")

# Use larger sample for realistic evaluation
sample_size = min(SAMPLE_SIZE, len(movie_features_full))
movie_features = movie_features_full.sample(n=sample_size, random_state=RANDOM_SEED).reset_index(drop=True)

model = ExistingUserRecommender(movie_features)
random.seed(RANDOM_SEED)


# -------------------------
# HELPERS
# -------------------------
title_to_genres = {}

for row in movie_features[["title", "genres"]].itertuples(index=False):
    if not isinstance(row.title, str) or not row.title.strip():
        continue

    if not isinstance(row.genres, str) or not row.genres.strip():
        title_to_genres[row.title] = set()
        continue

    title_to_genres[row.title] = {g.strip() for g in row.genres.split("|") if g.strip()}


def get_genres(title):
    return title_to_genres.get(title, set())


def relevance_score(rec_movie, test_movies):
    """Graded relevance based on max Jaccard overlap with held-out genres."""
    rec_genres = get_genres(rec_movie)

    if not rec_genres:
        return 0.0

    best = 0.0

    for t in test_movies:
        test_genres = get_genres(t)
        if not test_genres:
            continue

        union = rec_genres.union(test_genres)
        if not union:
            continue

        overlap = len(rec_genres.intersection(test_genres)) / len(union)
        if overlap > best:
            best = overlap

    return best


def is_relevant_binary(rec_movie, test_movies):
    return relevance_score(rec_movie, test_movies) > 0

    return False


def precision_at_k(recommended, test_movies, k):
    recommended = recommended[:k]
    if k <= 0:
        return 0

    hits = sum(1 for m in recommended if is_relevant_binary(m, test_movies))
    return hits / k


def recall_at_k(recommended, test_movies, k):
    if not test_movies:
        return 0

    found = 0

    for t in test_movies:
        t_genres = get_genres(t)

        for m in recommended:
            if t_genres.intersection(get_genres(m)):
                found += 1
                break

    return found / len(test_movies)


# -------------------------
# CREATE USERS FROM FULL DATASET
# -------------------------
all_movies = movie_features_full["title"].dropna().tolist()

users = []

for _ in range(NUM_USERS):
    history = random.sample(all_movies, MIN_HISTORY)
    users.append(history)


# -------------------------
# EVALUATION
# -------------------------
precision_scores = []
recall_scores = []
ndcg_scores = []
coverage_scores = []
evaluated_users = 0

start = time.time()

for i, history in enumerate(users):
    print(f"Processing user {i+1}/{NUM_USERS}")

    train = history[:3]
    test = history[3:]

    recs = model.recommend(train, mood="happy", top_n=TOP_K)
    if recs is None or recs.empty:
        continue

    recommended_titles = recs["title"].tolist()
    if not recommended_titles:
        continue

    relevance_scores = [relevance_score(title, test) for title in recommended_titles[:TOP_K]]

    p = precision_at_k(recommended_titles, test, TOP_K)
    r = recall_at_k(recommended_titles, test, TOP_K)
    n = ndcg_at_k_from_scores(relevance_scores, TOP_K)
    c = genre_coverage(recs)

    precision_scores.append(p)
    recall_scores.append(r)
    ndcg_scores.append(n)
    coverage_scores.append(c)
    evaluated_users += 1


# -------------------------
# FINAL RESULTS
# -------------------------
print("\n========== FINAL RESULTS ==========")
if evaluated_users == 0:
    print("No valid users were evaluated. Please check dataset/model configuration.")
else:
    print(f"Evaluated users: {evaluated_users}/{NUM_USERS}")
    print(f"Average Precision@{TOP_K}: {round(sum(precision_scores)/evaluated_users, 3)}")
    print(f"Average Recall@{TOP_K}: {round(sum(recall_scores)/evaluated_users, 3)}")
    print(f"Average NDCG@{TOP_K}: {round(sum(ndcg_scores)/evaluated_users, 3)}")
    print(f"Average Genre Coverage: {round(sum(coverage_scores)/evaluated_users, 2)}")

print(f"\nTime taken: {round(time.time() - start, 2)} seconds")


# -------------------------
# PLOTS
# -------------------------
plt.figure()
plt.plot(precision_scores, label=f"Precision@{TOP_K}")
plt.plot(recall_scores, label=f"Recall@{TOP_K}")
plt.plot(ndcg_scores, label=f"NDCG@{TOP_K}")
plt.legend()
plt.title("Model Performance (Full Dataset Evaluation)")
plt.xlabel("Users")
plt.ylabel("Score")
plt.grid()

plt.savefig("full_precision_recall.png")
plt.show()


plt.figure()
plt.plot(coverage_scores, label="Genre Coverage", color="green")
plt.legend()
plt.title("Genre Diversity")
plt.xlabel("Users")
plt.ylabel("Unique Genres")
plt.grid()

plt.savefig("full_coverage.png")
plt.show()