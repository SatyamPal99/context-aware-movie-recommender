import re
from rapidfuzz import fuzz, process


def normalize_title(title: str) -> str:
    """
    Lowercase, remove year in brackets, remove punctuation, normalize spaces.
    Example: 'Toy Story (1995)' -> 'toy story'
    """
    title = title.lower().strip()
    title = re.sub(r"\(\d{4}\)", "", title)   # remove year
    title = re.sub(r"[^a-z0-9\s]", " ", title)  # remove punctuation
    title = re.sub(r"\s+", " ", title).strip()  # normalize spaces
    return title


def find_closest_movie(user_input: str, movie_titles: list) -> str:
    """
    Improved movie matching:
    1. Exact normalized match
    2. Substring match
    3. Fuzzy match with token-based scorer
    """
    if not user_input or not user_input.strip():
        raise ValueError("Movie name cannot be empty.")

    normalized_input = normalize_title(user_input)

    # Build lookup
    normalized_to_original = {}
    normalized_titles = []

    for title in movie_titles:
        norm = normalize_title(title)
        normalized_titles.append(norm)
        if norm not in normalized_to_original:
            normalized_to_original[norm] = title

    # 1. Exact normalized match
    if normalized_input in normalized_to_original:
        return normalized_to_original[normalized_input]

    # 2. Substring match
    substring_matches = [
        original
        for norm, original in normalized_to_original.items()
        if normalized_input in norm
    ]

    if substring_matches:
        # Prefer shortest/best direct containment match
        substring_matches = sorted(substring_matches, key=len)
        return substring_matches[0]

    # 3. Fuzzy token-based match
    best_match = process.extractOne(
        normalized_input,
        normalized_titles,
        scorer=fuzz.token_sort_ratio
    )

    if not best_match:
        raise ValueError("Movie not found.")

    matched_norm, score, _ = best_match

    if score < 60:
        raise ValueError("Movie not found. Try a clearer name.")

    return normalized_to_original[matched_norm]