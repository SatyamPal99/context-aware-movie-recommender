## Movie Recommendation System

Hybrid-style movie recommender with Streamlit UI, content-based models, and optional Spark ALS pipeline.

### Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```
pip install -r req.txt
```

### Data

Place MovieLens-style files here:

- data/raw/movies.csv
- data/raw/ratings.csv

Generate movie feature table:

```
python src/run/run_feature_engineering.py
```

### Run Streamlit App

```
streamlit run app.py
```

### Spark ALS Pipeline (Optional)

Run ALS training + evaluation and export recommendations:

```
python src/spark/run_als.py --ratings data/raw/ratings.csv --movies data/raw/movies.csv
```

This outputs recommendations to:

```
data/processed/als_recommendations/
```

Then you can enable the Spark ALS toggle in the Streamlit UI for existing users.
