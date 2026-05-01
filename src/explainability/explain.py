def generate_explanation(
    movie_genres: str,
    mood: str,
    recent_movies: list,
    recommended_title: str,
    movie_features_df,
    preferred_genres: list = None,
    liked_movies: list = None,
    explanation_mode: str = "balanced",
    max_genres: int = 2,
    max_liked: int = 2,
    rank_position: int | None = None
):
    reasons = []

    movie_genres = movie_genres if isinstance(movie_genres, str) else ""
    movie_genres_lower = movie_genres.lower()
    recent_movies = recent_movies or []
    liked_movies = liked_movies or []

    mode = explanation_mode
    genres_limit = max_genres
    liked_limit = max_liked

    if explanation_mode == "balanced" and rank_position is not None:
        mode_cycle = ["genre", "liked", "both", "history", "mood", "popular"]
        mode = mode_cycle[(rank_position - 1) % len(mode_cycle)]
        genres_limit = 1 + (rank_position % 3)
        liked_limit = 1 + ((rank_position + 1) % 3)

    # -------------------------
    # GET MOVIE ROW
    # -------------------------
    movie_row = movie_features_df[
        movie_features_df["title"] == recommended_title
    ]

    popularity_score = 0
    if not movie_row.empty:
        popularity_score = movie_row.iloc[0].get("popularity_score", 0) or 0

    # -------------------------
    # ⭐ MUST WATCH LOGIC
    # -------------------------
    matched_pref = []
    if preferred_genres:
        matched_pref = [g for g in preferred_genres if g and g.lower() in movie_genres_lower]
        if matched_pref and mode in {"balanced", "genre", "both", "popular"}:
            reasons.append(f"matches your preferred genres ({', '.join(matched_pref[:genres_limit])})")

    if liked_movies:
        matched_liked = []
        movie_genre_set = {g.strip() for g in movie_genres_lower.split("|") if g.strip()}
        overlap_scored = []

        for liked_title in liked_movies[:10]:
            liked_row = movie_features_df[movie_features_df["title"] == liked_title]
            if liked_row.empty:
                continue

            liked_genre_set = {
                g.strip() for g in str(liked_row.iloc[0].get("genres", "") or "").lower().split("|") if g.strip()
            }

            overlap = len(movie_genre_set.intersection(liked_genre_set))
            if overlap > 0:
                overlap_scored.append((liked_title, overlap))

        if overlap_scored and mode in {"balanced", "liked", "both", "history"}:
            overlap_scored.sort(key=lambda x: x[1], reverse=True)
            matched_liked = [x[0] for x in overlap_scored[:liked_limit]]
            if len(matched_liked) == 1:
                reasons.append(f"similar to your liked movie '{matched_liked[0]}'")
            else:
                liked_text = "', '".join(matched_liked)
                reasons.append(f"similar to your liked movies '{liked_text}'")

    # -------------------------
    # HISTORY-BASED
    # -------------------------
    if recent_movies and mode in {"balanced", "history", "liked"}:
        matched_movie = None
        for candidate_movie in reversed(recent_movies[-3:]):
            candidate_row = movie_features_df[movie_features_df["title"] == candidate_movie]
            if candidate_row.empty:
                continue

            candidate_genres = str(candidate_row.iloc[0].get("genres", "") or "").lower()
            if any(g and g in movie_genres_lower for g in candidate_genres.split("|")):
                matched_movie = candidate_movie
                break

        if matched_movie and not reasons:
            reasons.append(f"similar to '{matched_movie}'")

    # -------------------------
    # MOOD-BASED
    # -------------------------
    mood_map = {
        "happy": ["comedy", "animation", "family"],
        "serious": ["drama", "crime"],
        "excited": ["action", "thriller", "adventure"]
    }

    if mood in mood_map and not reasons and mode in {"balanced", "mood", "history"}:
        if any(g in movie_genres_lower for g in mood_map[mood]):
            reasons.append(f"matches your {mood} mood")

    if (
        (not reasons and popularity_score > 15)
        or (mode == "popular" and popularity_score > 8)
    ):
        reasons.append("a highly rated pick you might like")

    # -------------------------
    # GENRE FALLBACK
    # -------------------------
    if not reasons:
        if movie_genres:
            main_genre = movie_genres.split("|")[0]
            reasons.append(f"belongs to {main_genre} genre")

    return "Recommended because it " + ", ".join(reasons) + "."