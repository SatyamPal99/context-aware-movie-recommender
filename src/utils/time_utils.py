from datetime import datetime


def get_time_of_day():
    """
    Returns time of day based on local system time.
    No external API needed (reliable).
    """

    hour = datetime.now().hour

    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"