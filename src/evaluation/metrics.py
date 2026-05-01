import math


# -------------------------
# PRECISION@K
# -------------------------
def precision_at_k(recommended, relevant, k=10):
    if k <= 0:
        return 0

    recommended_k = recommended[:k]

    hits = len(set(recommended_k) & set(relevant))

    return hits / k


# -------------------------
# RECALL@K
# -------------------------
def recall_at_k(recommended, relevant, k=10):
    if k <= 0:
        return 0

    recommended_k = recommended[:k]

    hits = len(set(recommended_k) & set(relevant))

    return hits / len(relevant) if relevant else 0


# -------------------------
# NDCG@K (BINARY RELEVANCE)
# -------------------------
def ndcg_at_k(recommended, relevant, k=10):
    if k <= 0:
        return 0

    recommended_k = recommended[:k]
    relevant_set = set(relevant)

    dcg = 0.0
    for rank, item in enumerate(recommended_k, start=1):
        if item in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(k, len(relevant_set))
    if ideal_hits == 0:
        return 0

    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


# -------------------------
# NDCG@K (GRADED RELEVANCE)
# -------------------------
def ndcg_at_k_from_scores(relevance_scores, k=10):
    if k <= 0:
        return 0

    scores_k = [float(s) for s in relevance_scores[:k]]
    if not scores_k:
        return 0

    dcg = sum(score / math.log2(rank + 1) for rank, score in enumerate(scores_k, start=1))

    ideal_scores = sorted(scores_k, reverse=True)
    idcg = sum(score / math.log2(rank + 1) for rank, score in enumerate(ideal_scores, start=1))

    return dcg / idcg if idcg > 0 else 0


# -------------------------
# COVERAGE (DIVERSITY)
# -------------------------
def genre_coverage(recommended_df):
    if recommended_df is None or recommended_df.empty or "genres" not in recommended_df.columns:
        return 0

    genres = []

    for g in recommended_df["genres"]:
        if not isinstance(g, str) or not g.strip():
            continue

        genres.extend(part.strip() for part in g.split("|") if part.strip())

    unique_genres = set(genres)

    return len(unique_genres)