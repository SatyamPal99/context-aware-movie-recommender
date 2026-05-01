from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"


def load_movies(data_dir: str | Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    path = Path(data_dir) / "movies.csv"
    return pd.read_csv(path)


def load_ratings(data_dir: str | Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    path = Path(data_dir) / "ratings.csv"
    return pd.read_csv(path)