import time
from pathlib import Path

import pandas as pd
import streamlit as st
from rapidfuzz import fuzz

from src.models.new_user_recommender import NewUserRecommender
from src.models.existing_user_recommender import ExistingUserRecommender

from src.utils.time_utils import get_time_of_day
from src.explainability.explain import generate_explanation
from src.realtime.update_model import (
    add_clicked_movie,
    add_liked_movie,
    get_movie_like_totals,
    get_recent_history,
    get_recent_history_entries,
    get_top_clicked_movie,
    user_exists,
    create_user,
    get_als_user_id,
    get_user_preferences,
    update_user_preferences,
    add_watch_later_movie,
    get_watch_later_movies,
    remove_watch_later_movie,
)

PROJECT_ROOT = Path(__file__).resolve().parent
ALS_RECS_DIR = PROJECT_ROOT / "data" / "processed" / "als_recommendations"
ALS_RECS_FILE = PROJECT_ROOT / "data" / "processed" / "als_recommendations.csv"
TOP_N = 15


# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# -------------------------
# LOAD DATA (CACHED)
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/movie_features.csv")


@st.cache_data
def load_als_recommendations():
    if ALS_RECS_FILE.exists():
        return pd.read_csv(ALS_RECS_FILE)

    if not ALS_RECS_DIR.exists():
        return pd.DataFrame()

    part_files = sorted(ALS_RECS_DIR.glob("part-*.csv"))
    if not part_files:
        return pd.DataFrame()

    dfs = [pd.read_csv(p) for p in part_files]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# -------------------------
# LOAD MODELS (CACHED)
# -------------------------
@st.cache_resource
def load_models(movie_features, model_cache_key: str = "v2-negative-genres"):
    return (
        NewUserRecommender(movie_features),
        ExistingUserRecommender(movie_features)
    )


movie_features = load_data()
als_recs = load_als_recommendations()
new_user_model, existing_user_model = load_models(movie_features, model_cache_key="v2-negative-genres")


def build_explanations(recs, mood, recent_movies, preferred_genres=None, liked_movies=None):
    """Build a stable explanation map for the current recommendation list."""
    explanations = {}
    for rank, row in enumerate(recs.head(TOP_N).itertuples(), 1):
        explanations[row.title] = generate_explanation(
            movie_genres=row.genres,
            mood=mood,
            recent_movies=recent_movies,
            recommended_title=row.title,
            movie_features_df=movie_features,
            preferred_genres=preferred_genres,
            liked_movies=liked_movies,
            rank_position=rank
        )
    return explanations


# -------------------------
# HELPERS
# -------------------------
def animated_progress(message: str):
    with st.spinner(message):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.003)
            progress.progress(i + 1)


def show_history_section(user_id: str):
    recent_entries = get_recent_history_entries(user_id)

    st.subheader("📺 Previously Watched")

    if recent_entries:
        for idx, item in enumerate(reversed(recent_entries), 1):
            movie = item.get("title") if isinstance(item, dict) else str(item)
            watched_at = item.get("watched_at") if isinstance(item, dict) else None
            click_count = item.get("clicks") if isinstance(item, dict) else None

            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                watched_label = watched_at if watched_at else "unknown time"
                clicks_label = int(click_count) if click_count is not None else 1
                st.markdown(
                    f"""
                    <div style="padding:10px; margin:5px 0; border-radius:10px; background-color:#262730;">
                        🎬 <b>{movie}</b><br>
                        ⏱ Watched at: {watched_label}<br>
                        🖱 Number of times watched: {clicks_label}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col2:
                if st.button("🔁 Watch Again", key=f"watch_again_{user_id}_{idx}"):
                    add_clicked_movie(user_id, movie)
                    st.success(f"Added '{movie}' to your history again.")
                    st.rerun()
            with col3:
                if st.button("❤️ Like", key=f"history_like_{user_id}_{idx}"):
                    add_liked_movie(user_id, movie)
                    st.success(f"Added '{movie}' to liked movies.")
                    st.rerun()
    else:
        st.info("No watch history yet. Start watching to build your profile!")

    recent_movies = [item.get("title") for item in recent_entries if isinstance(item, dict) and item.get("title")]
    if not recent_movies:
        recent_movies = get_recent_history(user_id)
    return recent_movies


def rerank_with_time_history(
    recs: pd.DataFrame,
    user_id: str,
    current_time_slot: str,
    movie_df: pd.DataFrame,
):
    if recs is None or recs.empty:
        return recs

    history_entries = get_recent_history_entries(user_id)
    if not history_entries:
        return recs

    same_slot_titles = [
        h.get("title")
        for h in history_entries
        if isinstance(h, dict) and h.get("time_of_day") == current_time_slot and h.get("title")
    ]
    if not same_slot_titles:
        return recs

    slot_movies = movie_df[movie_df["title"].isin(same_slot_titles)]
    if slot_movies.empty:
        return recs

    preferred_slot_genres = set()
    for genres in slot_movies["genres"].fillna(""):
        for g in str(genres).split("|"):
            g = g.strip().lower()
            if g:
                preferred_slot_genres.add(g)

    if not preferred_slot_genres:
        return recs

    reranked = recs.copy()
    bonus = []
    for row in reranked.itertuples():
        row_genres = {x.strip().lower() for x in str(getattr(row, "genres", "") or "").split("|") if x.strip()}
        bonus.append(0.12 if row_genres.intersection(preferred_slot_genres) else 0.0)

    reranked["final_score"] = pd.to_numeric(reranked["final_score"], errors="coerce").fillna(0.0) + pd.Series(bonus)
    return reranked.sort_values("final_score", ascending=False).head(len(recs))


def show_watch_later_section(user_id: str):
    st.subheader("🕒 Watch Later")
    watch_later_movies = get_watch_later_movies(user_id)

    if not watch_later_movies:
        st.info("No movies in your watch later playlist yet.")
        return

    for idx, title in enumerate(watch_later_movies, 1):
        col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
        with col1:
            st.markdown(
                f"""
                <div style="padding:10px; margin:5px 0; border-radius:10px; background-color:#1f3a5f;">
                    🎞 {title}
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            if st.button("▶ Watch Now", key=f"watch_later_watch_{user_id}_{idx}"):
                add_clicked_movie(user_id, title)
                remove_watch_later_movie(user_id, title)
                st.success(f"Started watching '{title}'.")
                st.rerun()
        with col3:
            if st.button("❌ Remove", key=f"watch_later_remove_{user_id}_{idx}"):
                remove_watch_later_movie(user_id, title)
                st.success(f"Removed '{title}' from watch later.")
                st.rerun()
        with col4:
            if st.button("❤️ Like", key=f"watch_later_like_{user_id}_{idx}"):
                add_liked_movie(user_id, title)
                st.success(f"Added '{title}' to liked movies.")
                st.rerun()


def get_als_recommendations(als_df: pd.DataFrame, user_id: str, top_n: int):
    if als_df.empty:
        return None

    try:
        user_id_int = int(user_id)
    except (ValueError, TypeError):
        return None

    user_recs = als_df[als_df["userId"] == user_id_int]
    if user_recs.empty:
        return None

    user_recs = user_recs.sort_values("score", ascending=False).head(top_n)
    user_recs = user_recs.rename(columns={"score": "final_score"})
    return user_recs


@st.cache_data
def build_als_profiles(als_df: pd.DataFrame):
    if als_df.empty:
        return pd.DataFrame()

    df = als_df.copy()
    df["score"] = pd.to_numeric(df.get("score"), errors="coerce").fillna(0.0)
    df = df.dropna(subset=["userId", "title"])

    profiles = []

    for user_id, group in df.groupby("userId"):
        title_score_map = {}
        genre_score_map = {}

        for row in group.itertuples():
            title = str(row.title).strip().lower()
            score = float(row.score)

            if title:
                title_score_map[title] = max(score, title_score_map.get(title, 0.0))

            genres = str(getattr(row, "genres", "") or "")
            for genre in genres.split("|"):
                genre = genre.strip().lower()
                if genre and genre != "nan":
                    genre_score_map[genre] = genre_score_map.get(genre, 0.0) + max(score, 0.0)

        profiles.append({
            "userId": int(user_id),
            "title_score_map": title_score_map,
            "genre_score_map": genre_score_map,
            "avg_score": float(group["score"].mean()) if not group.empty else 0.0,
        })

    return pd.DataFrame(profiles)


def map_als_user_id(preferred_genres, liked_movies, profiles: pd.DataFrame):
    if profiles.empty:
        return None

    preferred_set = {g.strip().lower() for g in (preferred_genres or []) if g}
    liked_set = {m.strip().lower() for m in (liked_movies or []) if m}

    max_avg_score = max(float(profiles["avg_score"].max()), 1e-9)

    best_id = None
    best_score = -1.0

    for row in profiles.itertuples():
        title_score_map = row.title_score_map or {}
        genre_score_map = row.genre_score_map or {}

        movie_similarity = 0.0
        if liked_set and title_score_map:
            per_movie = []
            for liked_title in liked_set:
                best_match = 0.0
                for candidate_title, pred_score in title_score_map.items():
                    sim = fuzz.token_set_ratio(liked_title, candidate_title) / 100.0
                    weighted = sim * max(float(pred_score), 0.0)
                    if weighted > best_match:
                        best_match = weighted
                per_movie.append(best_match)
            movie_similarity = sum(per_movie) / len(per_movie)

        genre_overlap = 0.0
        if preferred_set and genre_score_map:
            matched_weight = sum(genre_score_map.get(g, 0.0) for g in preferred_set)
            total_weight = sum(genre_score_map.values())
            if total_weight > 0:
                genre_overlap = matched_weight / total_weight

        score_strength = float(row.avg_score) / max_avg_score

        score = (0.55 * movie_similarity) + (0.35 * genre_overlap) + (0.10 * score_strength)

        if score > best_score:
            best_score = score
            best_id = row.userId

    return int(best_id) if best_score > 0 else None


def _genres_for_titles(titles, movie_df):
    genre_counter = {}
    title_set = set(titles or [])
    if not title_set:
        return genre_counter

    rows = movie_df[movie_df["title"].isin(title_set)]
    for g in rows["genres"].fillna(""):
        for item in str(g).split("|"):
            item = item.strip().lower()
            if item:
                genre_counter[item] = genre_counter.get(item, 0) + 1
    return genre_counter


def rerank_als_recommendations(recs: pd.DataFrame, recent_movies, preferred_genres, liked_movies, negative_genres, movie_df):
    if recs is None or recs.empty:
        return recs

    negative_set = {g.strip().lower() for g in (negative_genres or []) if g}

    if negative_set:
        def has_negative(genres):
            row_set = {x.strip().lower() for x in str(genres or "").split("|") if x.strip()}
            return bool(row_set.intersection(negative_set))

        recs = recs[~recs["genres"].apply(has_negative)].copy()
        if recs.empty:
            return recs

    reranked = recs.copy()
    max_score = reranked["final_score"].max() if len(reranked) else 0
    if max_score > 0:
        reranked["als_norm"] = reranked["final_score"] / max_score
    else:
        reranked["als_norm"] = 0

    profile_genres = _genres_for_titles(recent_movies, movie_df)
    liked_genres = _genres_for_titles(liked_movies, movie_df)
    for g, c in liked_genres.items():
        profile_genres[g] = profile_genres.get(g, 0) + c
    for g in preferred_genres or []:
        g = str(g).strip().lower()
        if g:
            profile_genres[g] = profile_genres.get(g, 0) + 2

    recent_lower = [str(t).lower() for t in (recent_movies or [])]
    adjusted = []
    for row in reranked.itertuples():
        row_genres = {x.strip().lower() for x in str(getattr(row, "genres", "") or "").split("|") if x.strip()}
        genre_fit = 0.0
        if profile_genres:
            matched = sum(profile_genres.get(g, 0) for g in row_genres)
            total = sum(profile_genres.values())
            genre_fit = matched / total if total > 0 else 0.0

        title = str(row.title).lower()
        duplicate_penalty = 0.0
        if recent_lower:
            max_sim = max(fuzz.token_set_ratio(title, x) / 100.0 for x in recent_lower)
            duplicate_penalty = max(0.0, max_sim - 0.75)

        negative_penalty = 1.0 if row_genres.intersection(negative_set) else 0.0

        final = (0.75 * row.als_norm) + (0.30 * genre_fit) - (0.20 * duplicate_penalty) - (0.80 * negative_penalty)
        adjusted.append(final)

    reranked["final_score"] = adjusted
    return reranked.sort_values("final_score", ascending=False).head(len(recs))


def _normalize_scores(df: pd.DataFrame, score_col: str = "final_score") -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df[score_col], errors="coerce").fillna(0)
    s_min, s_max = s.min(), s.max()
    if s_max - s_min < 1e-9:
        return pd.Series([0.5] * len(df), index=df.index)
    return (s - s_min) / (s_max - s_min)


def blend_default_and_als(default_recs: pd.DataFrame, als_recs_df: pd.DataFrame | None, top_n: int) -> pd.DataFrame:
    """Blend default and ALS recommendations into one ranked list without exposing ALS to users."""
    if default_recs is None or default_recs.empty:
        return als_recs_df.head(top_n) if als_recs_df is not None else pd.DataFrame()

    default_df = default_recs.copy()
    default_df["default_norm"] = _normalize_scores(default_df)
    default_df["als_norm"] = 0.0

    if als_recs_df is not None and not als_recs_df.empty:
        als_df = als_recs_df.copy()
        als_df["als_norm"] = _normalize_scores(als_df)
        als_df["default_norm"] = 0.0

        needed = ["movieId", "title", "genres", "default_norm", "als_norm"]
        merged = pd.concat([
            default_df[needed],
            als_df[needed],
        ], ignore_index=True)

        merged = (
            merged.groupby(["movieId", "title", "genres"], as_index=False)
            .agg(default_norm=("default_norm", "max"), als_norm=("als_norm", "max"))
        )
    else:
        merged = default_df[["movieId", "title", "genres", "default_norm", "als_norm"]].copy()

    merged["final_score"] = (0.65 * merged["default_norm"]) + (0.35 * merged["als_norm"])
    return merged.sort_values("final_score", ascending=False).head(top_n)


def mix_with_highly_rated(
    recs: pd.DataFrame,
    movie_df: pd.DataFrame,
    top_n: int,
    recent_movies=None,
    liked_movies=None,
    negative_genres=None,
):
    """Blend a few highly-rated popular titles into personalized recommendations."""
    if recs is None or recs.empty:
        return recs

    recs = recs.copy()
    recs["rec_origin"] = recs.get("rec_origin", "personalized")

    negative_set = {g.strip().lower() for g in (negative_genres or []) if g}
    excluded_titles = set((recent_movies or [])) | set((liked_movies or [])) | set(recs["title"].tolist())

    candidates = movie_df.copy()
    candidates = candidates[~candidates["title"].isin(excluded_titles)]

    if negative_set:
        def has_negative(genres):
            row_set = {x.strip().lower() for x in str(genres or "").split("|") if x.strip()}
            return bool(row_set.intersection(negative_set))

        candidates = candidates[~candidates["genres"].apply(has_negative)]

    if candidates.empty:
        return recs.sort_values("final_score", ascending=False).head(top_n)

    candidates["popularity_norm"] = candidates["popularity_score"] / max(candidates["popularity_score"].max(), 1e-9)
    high_n = min(max(2, top_n // 5), top_n)
    popular = candidates.sort_values(["popularity_norm", "avg_rating"], ascending=False).head(high_n).copy()

    if popular.empty:
        return recs.sort_values("final_score", ascending=False).head(top_n)

    score_floor = recs["final_score"].min() if len(recs) else 0
    popular["final_score"] = (0.85 * score_floor) + (0.15 * popular["popularity_norm"])
    popular["rec_origin"] = "you_might_like"

    needed_cols = ["movieId", "title", "genres", "final_score", "rec_origin"]
    mixed = pd.concat([recs[needed_cols], popular[needed_cols]], ignore_index=True)
    mixed = mixed.sort_values("final_score", ascending=False).head(top_n)
    return mixed


als_profiles = build_als_profiles(als_recs)


# -------------------------
# WELCOME SECTION
# -------------------------
st.markdown("""
# 🎬 Welcome to Movie Recommendation System

### 👨‍💻 Developed by:
- S Kartik Iyer
- Ravi Sharma
- Satyam Pal

---
""")

st.success("🚀 Smart Recommendation System Ready")

mood = st.selectbox("Select your mood", ["happy", "serious", "excited"])
time_of_day = get_time_of_day()
st.write(f"🕒 Detected Time: **{time_of_day}**")

user_type = st.radio("Are you a new user?", ["Yes", "No"], horizontal=True)

all_genres = sorted(list(set("|".join(movie_features["genres"].fillna("")).split("|"))))
all_genres = [g for g in all_genres if g.strip()]


# -------------------------
# NEW USER FLOW
# -------------------------
if user_type == "Yes":
    st.subheader("🆕 New User Setup")

    new_user_id = st.text_input("Create a user ID for future use")

    selected_genres = st.multiselect(
        "Select genres (first is used to filter movie choices)",
        options=all_genres
    )

    selected_genre = selected_genres[0] if selected_genres else all_genres[0]
    preferred_genres = selected_genres[1:] if len(selected_genres) > 1 else []

    filtered_movies = movie_features[
        movie_features["genres"].str.contains(selected_genre, na=False)
    ]["title"].dropna().tolist()

    liked_movies = st.multiselect(
        "Select movies you like",
        options=filtered_movies[:200]
    )

    negative_genres = st.multiselect(
        "Select genres you do NOT want",
        options=all_genres
    )

    if st.button("Get Recommendations", key="new_user_recs"):
        if not new_user_id.strip():
            st.error("❌ Please create a valid user ID.")
            st.stop()

        if user_exists(new_user_id.strip()):
            st.error("❌ This user ID already exists. Please choose a different one.")
            st.stop()

        als_user_id_value = map_als_user_id(preferred_genres, liked_movies, als_profiles) if not als_profiles.empty else None
        create_user(
            new_user_id.strip(),
            liked_movies=liked_movies,
            preferred_genres=preferred_genres,
            disliked_genres=negative_genres,
            als_user_id=als_user_id_value
        )
        animated_progress("🎬 Generating recommendations...")

        default_recs = new_user_model.recommend(
            preferred_genres=preferred_genres,
            liked_movies=liked_movies,
            mood=mood,
            top_n=TOP_N,
            negative_genres=negative_genres
        )

        als_result = None
        if als_user_id_value is not None:
            als_result = get_als_recommendations(als_recs, als_user_id_value, top_n=TOP_N)
            if als_result is not None and not als_result.empty:
                als_result = rerank_als_recommendations(
                    recs=als_result,
                    recent_movies=[],
                    preferred_genres=preferred_genres,
                    liked_movies=liked_movies,
                    negative_genres=negative_genres,
                    movie_df=movie_features
                )

        recs = blend_default_and_als(default_recs, als_result, top_n=TOP_N)
        recs = mix_with_highly_rated(
            recs=recs,
            movie_df=movie_features,
            top_n=TOP_N,
            liked_movies=liked_movies,
            negative_genres=negative_genres
        )

        st.session_state["recs"] = recs
        st.session_state["user_id"] = new_user_id.strip()
        st.session_state["rec_source"] = "Personalized Hybrid Model"
        st.session_state["explanations"] = build_explanations(
            recs=recs,
            mood=mood,
            recent_movies=[],
            preferred_genres=preferred_genres,
            liked_movies=liked_movies
        )
        st.session_state["preferred_genres"] = preferred_genres
        st.session_state["disliked_genres"] = negative_genres
        st.session_state["liked_movies"] = liked_movies

        st.success(f"✅ User '{new_user_id.strip()}' created successfully.")


# -------------------------
# EXISTING USER FLOW
# -------------------------
else:
    st.subheader("👤 Existing User")

    existing_user_id = st.text_input("Enter your user ID")
    all_movie_titles = movie_features["title"].dropna().astype(str).tolist()

    if existing_user_id:
        existing_user_id = existing_user_id.strip()

        if not user_exists(existing_user_id):
            st.error("❌ User not registered. Please enter a valid user ID.")
            st.stop()

        recent_movies = show_history_section(existing_user_id)
        liked_movies, preferred_genres, disliked_genres = get_user_preferences(existing_user_id)

        with st.expander("⚙ Edit Profile Preferences", expanded=False):
            edited_preferred_genres = st.multiselect(
                "Edit preferred genres",
                options=all_genres,
                default=preferred_genres,
                key=f"edit_pref_genres_{existing_user_id}"
            )
            edited_liked_movies = st.multiselect(
                "Edit liked movies",
                options=all_movie_titles,
                default=liked_movies,
                key=f"edit_liked_movies_{existing_user_id}"
            )

            if st.button("Save Profile Changes", key=f"save_profile_{existing_user_id}"):
                cleaned_genres = list(dict.fromkeys([g for g in edited_preferred_genres if isinstance(g, str) and g.strip()]))
                cleaned_liked = list(dict.fromkeys([m for m in edited_liked_movies if isinstance(m, str) and m.strip()]))

                update_user_preferences(
                    existing_user_id,
                    liked_movies=cleaned_liked,
                    preferred_genres=cleaned_genres,
                )

                st.session_state["preferred_genres"] = cleaned_genres
                st.session_state["liked_movies"] = cleaned_liked
                st.success("✅ Profile updated successfully. Your next recommendations will use these changes.")
                st.rerun()

        st.subheader("❤️ Liked Movies")
        liked_search_text = st.text_input("Search in your liked movies", key=f"liked_search_{existing_user_id}")
        liked_filtered = [
            m for m in liked_movies
            if liked_search_text.strip().lower() in m.lower()
        ] if liked_search_text.strip() else liked_movies

        with st.container(height=220):
            if liked_filtered:
                for idx, title in enumerate(liked_filtered, 1):
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(f"• {title}")
                    with col2:
                        if st.button("▶ Watch", key=f"liked_watch_{existing_user_id}_{idx}"):
                            add_clicked_movie(existing_user_id, title)
                            st.success(f"Added '{title}' to your history.")
                            st.rerun()
            else:
                st.caption("No liked movies match your search.")

        fav_title, fav_clicks = get_top_clicked_movie(existing_user_id)
        st.subheader("⭐ Favourite Movie")
        if fav_title:
            st.info(f"{fav_title}  |  watched : {fav_clicks} times")
        else:
            st.caption("Favourite movie will appear once you start watching titles.")

        show_watch_later_section(existing_user_id)

        if st.button("Get Recommendations", key="existing_user_recs"):
            if not recent_movies:
                if not liked_movies and not preferred_genres:
                    st.warning("No history or preferences found.")
                    st.stop()

                st.info("Using your saved preferences...")
                default_recs = new_user_model.recommend(
                    preferred_genres=preferred_genres,
                    liked_movies=liked_movies,
                    mood=mood,
                    top_n=TOP_N,
                    negative_genres=disliked_genres
                )
            else:
                animated_progress("🎬 Updating your personalized recommendations...")
                default_recs = existing_user_model.recommend(
                    recent_movies=recent_movies,
                    mood=mood,
                    top_n=TOP_N,
                    negative_genres=disliked_genres
                )

            mapped_als_id = get_als_user_id(existing_user_id)
            if not mapped_als_id and not als_profiles.empty:
                mapped_als_id = map_als_user_id(preferred_genres, liked_movies, als_profiles)
                if mapped_als_id:
                    update_user_preferences(existing_user_id, als_user_id=mapped_als_id)

            als_result = get_als_recommendations(als_recs, mapped_als_id, top_n=TOP_N) if mapped_als_id is not None else None
            if als_result is not None and not als_result.empty:
                als_result = rerank_als_recommendations(
                    recs=als_result,
                    recent_movies=recent_movies,
                    preferred_genres=preferred_genres,
                    liked_movies=liked_movies,
                    negative_genres=disliked_genres,
                    movie_df=movie_features
                )

            combined = blend_default_and_als(default_recs, als_result, top_n=TOP_N)
            combined = mix_with_highly_rated(
                recs=combined,
                movie_df=movie_features,
                top_n=TOP_N,
                recent_movies=recent_movies,
                liked_movies=liked_movies,
                negative_genres=disliked_genres
            )
            combined = rerank_with_time_history(
                recs=combined,
                user_id=existing_user_id,
                current_time_slot=time_of_day,
                movie_df=movie_features,
            )

            st.session_state["recs"] = combined
            st.session_state["user_id"] = existing_user_id
            st.session_state["rec_source"] = "Personalized Hybrid Model"
            st.session_state["explanations"] = build_explanations(
                recs=combined,
                mood=mood,
                recent_movies=recent_movies,
                preferred_genres=preferred_genres,
                liked_movies=liked_movies
            )
            st.session_state["preferred_genres"] = preferred_genres
            st.session_state["disliked_genres"] = disliked_genres
            st.session_state["liked_movies"] = liked_movies


# -------------------------
# SHOW RECOMMENDATIONS
# -------------------------
if "recs" in st.session_state:
    st.subheader("🎬 Top Recommendations")
    if "rec_source" in st.session_state:
        st.caption(f"Source: {st.session_state['rec_source']}")

    recs = st.session_state["recs"]
    movie_like_totals = get_movie_like_totals()
    current_user_id = st.session_state.get("user_id", "")
    explanation_map = st.session_state.get("explanations", {})
    current_pref_genres = st.session_state.get("preferred_genres", [])
    current_liked_movies = st.session_state.get("liked_movies", [])

    for i, row in enumerate(recs.head(TOP_N).itertuples(), 1):
        explanation = explanation_map.get(
            row.title,
            generate_explanation(
                movie_genres=row.genres,
                mood=mood,
                recent_movies=get_recent_history(current_user_id),
                recommended_title=row.title,
                movie_features_df=movie_features,
                preferred_genres=current_pref_genres,
                liked_movies=current_liked_movies,
                rank_position=i
            )
        )

        st.markdown(f"### {i}. {row.title}")
        st.write(f"⭐ Score: {round(row.final_score, 3)}")
        total_likes = movie_like_totals.get(str(row.title), 0)
        st.write(f"👍 Total Likes (all users): {total_likes}")
        tags = str(getattr(row, "genres", "") or "")
        if tags and tags.lower() != "nan":
            st.caption(f"Tags: {tags.replace('|', ', ')}")
        st.write(f"💡 {explanation}")

        if st.button(f"🎥 Watch {row.title}", key=f"watch_{row.movieId}_{i}"):
            add_clicked_movie(current_user_id, row.title)
            st.success(f"Added '{row.title}' to your history.")
            st.balloons()
            st.rerun()

        if current_user_id and st.button(f"🕒 Watch Later: {row.title}", key=f"watch_later_{row.movieId}_{i}"):
            add_watch_later_movie(current_user_id, row.title)
            st.success(f"Added '{row.title}' to watch later.")
            st.rerun()

        if current_user_id and st.button(f"❤️ Like {row.title}", key=f"rec_like_{row.movieId}_{i}"):
            add_liked_movie(current_user_id, row.title)
            session_likes = st.session_state.get("liked_movies", [])
            if row.title not in session_likes:
                st.session_state["liked_movies"] = [*session_likes, row.title]
            st.success(f"Added '{row.title}' to liked movies.")
            st.rerun()