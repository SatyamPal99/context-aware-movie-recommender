import pandas as pd


def apply_context_boost(recommendations: pd.DataFrame, mood: str, time_of_day: str) -> pd.DataFrame:
    recommendations = recommendations.copy()
    recommendations["context_bonus"] = 0.0

    mood = mood.lower()
    time_of_day = time_of_day.lower()

    if mood == "happy":
        recommendations.loc[recommendations["genres"].str.contains("Comedy", na=False), "context_bonus"] += 0.2
    elif mood == "serious":
        recommendations.loc[recommendations["genres"].str.contains("Drama", na=False), "context_bonus"] += 0.2
    elif mood == "excited":
        recommendations.loc[recommendations["genres"].str.contains("Action", na=False), "context_bonus"] += 0.2

    if time_of_day == "night":
        recommendations.loc[recommendations["genres"].str.contains("Thriller|Horror", na=False), "context_bonus"] += 0.2

    recommendations["final_score"] = recommendations["final_score"] + recommendations["context_bonus"]
    return recommendations.sort_values("final_score", ascending=False)