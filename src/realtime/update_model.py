import json
import os
from datetime import datetime

USER_FILE = "data/processed/user_profiles.json"


def _infer_time_of_day_from_hour(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


def _ensure_user_schema(user_data: dict) -> dict:
    user_data = user_data or {}
    user_data.setdefault("recent_watch_history", [])
    user_data.setdefault("liked_movies", [])
    user_data.setdefault("preferred_genres", [])
    user_data.setdefault("disliked_genres", [])
    user_data.setdefault("als_user_id", None)
    user_data.setdefault("movie_clicks", {})
    user_data.setdefault("watch_later", [])

    normalized_history = []
    for item in user_data.get("recent_watch_history", []):
        if isinstance(item, str):
            normalized_history.append(
                {
                    "title": item,
                    "watched_at": None,
                    "time_of_day": None,
                    "clicks": int(user_data["movie_clicks"].get(item, 1)),
                }
            )
        elif isinstance(item, dict) and item.get("title"):
            title = str(item.get("title"))
            normalized_history.append(
                {
                    "title": title,
                    "watched_at": item.get("watched_at"),
                    "time_of_day": item.get("time_of_day"),
                    "clicks": int(item.get("clicks", user_data["movie_clicks"].get(title, 1))),
                }
            )

    user_data["recent_watch_history"] = normalized_history
    return user_data


def load_users():
    if not os.path.exists(USER_FILE):
        return {}

    try:
        with open(USER_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
            if not isinstance(raw, dict):
                return {}

            users = {}
            for uid, profile in raw.items():
                users[uid] = _ensure_user_schema(profile if isinstance(profile, dict) else {})
            return users
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def save_users(users):
    os.makedirs(os.path.dirname(USER_FILE), exist_ok=True)
    with open(USER_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4)


def user_exists(user_id: str) -> bool:
    users = load_users()
    return user_id in users


def create_user(user_id: str, liked_movies=None, preferred_genres=None, disliked_genres=None, als_user_id=None):
    users = load_users()

    if user_id not in users:
        users[user_id] = {
            "recent_watch_history": [],
            "liked_movies": liked_movies or [],
            "preferred_genres": preferred_genres or [],
            "disliked_genres": disliked_genres or [],
            "als_user_id": als_user_id,
            "movie_clicks": {},
            "watch_later": [],
        }

    save_users(users)

def get_user_preferences(user_id: str):
    users = load_users()

    if user_id not in users:
        return [], []

    liked = users[user_id].get("liked_movies", [])
    genres = users[user_id].get("preferred_genres", [])
    disliked = users[user_id].get("disliked_genres", [])

    return liked, genres, disliked


def get_als_user_id(user_id: str):
    users = load_users()

    if user_id not in users:
        return None

    return users[user_id].get("als_user_id")


def update_user_preferences(user_id: str, liked_movies=None, preferred_genres=None, disliked_genres=None, als_user_id=None):
    users = load_users()

    if user_id not in users:
        return

    if liked_movies is not None:
        users[user_id]["liked_movies"] = liked_movies
    if preferred_genres is not None:
        users[user_id]["preferred_genres"] = preferred_genres
    if disliked_genres is not None:
        users[user_id]["disliked_genres"] = disliked_genres
    if als_user_id is not None:
        users[user_id]["als_user_id"] = als_user_id

    save_users(users)

def add_clicked_movie(user_id: str, movie_title: str):
    users = load_users()

    if user_id not in users:
        users[user_id] = _ensure_user_schema({"recent_watch_history": []})

    users[user_id] = _ensure_user_schema(users[user_id])

    now = datetime.now()
    watched_at = now.isoformat(timespec="seconds")
    time_of_day = _infer_time_of_day_from_hour(now.hour)

    clicks = users[user_id].get("movie_clicks", {})
    clicks[movie_title] = int(clicks.get(movie_title, 0)) + 1
    users[user_id]["movie_clicks"] = clicks

    history = users[user_id].get("recent_watch_history", [])
    history = [h for h in history if not (isinstance(h, dict) and h.get("title") == movie_title)]
    history.append(
        {
            "title": movie_title,
            "watched_at": watched_at,
            "time_of_day": time_of_day,
            "clicks": clicks[movie_title],
        }
    )

    users[user_id]["recent_watch_history"] = history[-10:]

    if movie_title in users[user_id].get("watch_later", []):
        users[user_id]["watch_later"] = [m for m in users[user_id]["watch_later"] if m != movie_title]

    save_users(users)


def get_recent_history(user_id: str):
    users = load_users()

    if user_id not in users:
        return []

    entries = users[user_id].get("recent_watch_history", [])
    titles = []
    for item in entries:
        if isinstance(item, str):
            titles.append(item)
        elif isinstance(item, dict) and item.get("title"):
            titles.append(item["title"])
    return titles


def get_recent_history_entries(user_id: str):
    users = load_users()
    if user_id not in users:
        return []
    return users[user_id].get("recent_watch_history", [])


def get_top_clicked_movie(user_id: str):
    users = load_users()
    if user_id not in users:
        return None, 0

    clicks = users[user_id].get("movie_clicks", {})
    if not clicks:
        return None, 0

    title = max(clicks, key=lambda k: clicks[k])
    return title, int(clicks[title])


def add_watch_later_movie(user_id: str, movie_title: str):
    users = load_users()
    if user_id not in users:
        users[user_id] = _ensure_user_schema({})

    watch_later = users[user_id].get("watch_later", [])
    if movie_title not in watch_later:
        watch_later.append(movie_title)
    users[user_id]["watch_later"] = watch_later
    save_users(users)


def get_watch_later_movies(user_id: str):
    users = load_users()
    if user_id not in users:
        return []
    return users[user_id].get("watch_later", [])


def remove_watch_later_movie(user_id: str, movie_title: str):
    users = load_users()
    if user_id not in users:
        return

    users[user_id]["watch_later"] = [m for m in users[user_id].get("watch_later", []) if m != movie_title]
    save_users(users)


def add_liked_movie(user_id: str, movie_title: str):
    users = load_users()
    if user_id not in users:
        users[user_id] = _ensure_user_schema({})

    users[user_id] = _ensure_user_schema(users[user_id])
    liked_movies = users[user_id].get("liked_movies", [])
    if movie_title not in liked_movies:
        liked_movies.append(movie_title)

    users[user_id]["liked_movies"] = liked_movies
    save_users(users)


def get_movie_like_totals() -> dict[str, int]:
    """Return a case-insensitive aggregate of liked counts per movie title."""
    users = load_users()
    canonical_titles = {}
    like_totals = {}

    for profile in users.values():
        if not isinstance(profile, dict):
            continue

        for title in profile.get("liked_movies", []):
            title = str(title).strip()
            if not title:
                continue

            key = title.casefold()
            if key not in canonical_titles:
                canonical_titles[key] = title
            like_totals[key] = like_totals.get(key, 0) + 1

    return {canonical_titles[k]: v for k, v in like_totals.items()}


def get_movie_total_likes(movie_title: str) -> int:
    """Return total number of users who liked the given movie title."""
    normalized_title = str(movie_title or "").strip().casefold()
    if not normalized_title:
        return 0

    users = load_users()
    total = 0

    for profile in users.values():
        if not isinstance(profile, dict):
            continue

        liked_movies = {str(title).strip().casefold() for title in profile.get("liked_movies", []) if str(title).strip()}
        if normalized_title in liked_movies:
            total += 1

    return total
