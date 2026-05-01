import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col,
    row_number,
    count as spark_count,
    expr,
    size,
    array_intersect,
    avg,
    explode
)
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RATINGS = PROJECT_ROOT / "data" / "raw" / "ratings.csv"
DEFAULT_MOVIES = PROJECT_ROOT / "data" / "raw" / "movies.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "als_recommendations"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "saved_models" / "als"


def build_spark(app_name: str, driver_mem: str, executor_mem: str) -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.driver.memory", driver_mem)
        .config("spark.executor.memory", executor_mem)
        .config("spark.sql.shuffle.partitions", "50")
        .config("spark.default.parallelism", "50")
        .getOrCreate()
    )


def load_ratings(spark: SparkSession, path: Path, max_ratings: int | None):
    ratings = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(path))
    )

    required = {"userId", "movieId", "rating"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"ratings.csv missing columns: {sorted(missing)}")

    ratings = ratings.select(
        col("userId").cast("int").alias("userId"),
        col("movieId").cast("int").alias("movieId"),
        col("rating").cast("double").alias("rating"),
        *([col("timestamp").cast("long")] if "timestamp" in ratings.columns else [])
    )

    if max_ratings:
        ratings = ratings.orderBy(expr("rand()"))
        ratings = ratings.limit(max_ratings)

    return ratings


def load_movies(spark: SparkSession, path: Path):
    movies = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(path))
    )

    required = {"movieId", "title", "genres"}
    missing = required - set(movies.columns)
    if missing:
        raise ValueError(f"movies.csv missing columns: {sorted(missing)}")

    return movies.select(
        col("movieId").cast("int").alias("movieId"),
        col("title").cast("string").alias("title"),
        col("genres").cast("string").alias("genres")
    )


def filter_active_users(ratings, min_ratings: int):
    counts = ratings.groupBy("userId").agg(spark_count("rating").alias("cnt"))
    active = counts.filter(col("cnt") >= min_ratings).select("userId")
    return ratings.join(active, on="userId", how="inner")


def split_train_test(ratings, test_per_user: int, seed: int):
    if "timestamp" in ratings.columns:
        window = Window.partitionBy("userId").orderBy(col("timestamp").desc())
        ranked = ratings.withColumn("rn", row_number().over(window))
        test = ranked.filter(col("rn") <= test_per_user).drop("rn")
        train = ranked.filter(col("rn") > test_per_user).drop("rn")
    else:
        train, test = ratings.randomSplit([0.8, 0.2], seed)
        train_users = train.select("userId").distinct()
        test = test.join(train_users, on="userId", how="inner")

    return train, test


def train_als(train, rank: int, max_iter: int, reg_param: float):
    als = ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        nonnegative=True,
        coldStartStrategy="drop",
        implicitPrefs=False,
        rank=rank,
        maxIter=max_iter,
        regParam=reg_param
    )
    return als.fit(train)


def evaluate_model(model, test, k: int):
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    predictions = model.transform(test)
    rmse = evaluator.evaluate(predictions)

    actual = test.groupBy("userId").agg(expr("collect_set(movieId) as actual"))
    recs = model.recommendForAllUsers(k)
    recs = recs.select(
        "userId",
        expr("transform(recommendations, x -> x.movieId) as recs")
    )

    joined = actual.join(recs, on="userId", how="inner")
    scored = joined.select(
        col("userId"),
        size(array_intersect(col("actual"), col("recs"))).alias("hits"),
        size(col("actual")).alias("actual_size"),
        expr(
            "aggregate("
            "zip_with(recs, sequence(1, size(recs)), (m, i) -> IF(array_contains(actual, m), 1D / log2(i + 1), 0D)),"
            "0D,"
            "(acc, x) -> acc + x"
            ")"
        ).alias("dcg"),
        expr(
            "aggregate("
            "sequence(1, least(size(actual), {k})),"
            "0D,"
            "(acc, i) -> acc + 1D / log2(i + 1)"
            ")".format(k=k)
        ).alias("idcg")
    )

    scored = scored.select(
        (col("hits") / k).alias("precision"),
        expr("case when actual_size = 0 then 0 else hits / actual_size end").alias("recall"),
        expr("case when idcg = 0 then 0 else dcg / idcg end").alias("ndcg")
    )

    metrics = scored.agg(
        avg("precision").alias("precision_at_k"),
        avg("recall").alias("recall_at_k"),
        avg("ndcg").alias("ndcg_at_k")
    ).collect()[0]

    return rmse, metrics["precision_at_k"], metrics["recall_at_k"], metrics["ndcg_at_k"]


def export_recommendations(model, movies, output_dir: Path, top_n: int):
    recs = model.recommendForAllUsers(top_n)
    exploded = recs.select(
        col("userId"),
        explode(col("recommendations")).alias("rec")
    ).select(
        col("userId"),
        col("rec.movieId").alias("movieId"),
        col("rec.rating").alias("score")
    )

    enriched = exploded.join(movies, on="movieId", how="left")

    (
        enriched
        .orderBy(col("userId"), col("score").desc())
        .write
        .mode("overwrite")
        .option("header", True)
        .csv(str(output_dir))
    )


def export_recommendations_local(model, movies, output_file: Path, top_n: int):
    recs = model.recommendForAllUsers(top_n)
    exploded = recs.select(
        col("userId"),
        explode(col("recommendations")).alias("rec")
    ).select(
        col("userId"),
        col("rec.movieId").alias("movieId"),
        col("rec.rating").alias("score")
    )

    enriched = exploded.join(movies, on="movieId", how="left")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    pdf = enriched.orderBy(col("userId"), col("score").desc()).toPandas()
    pdf.to_csv(output_file, index=False)


def main():
    parser = argparse.ArgumentParser(description="Train Spark ALS and export recommendations.")
    parser.add_argument("--ratings", default=str(DEFAULT_RATINGS), help="Path to ratings.csv")
    parser.add_argument("--movies", default=str(DEFAULT_MOVIES), help="Path to movies.csv")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output directory for ALS recs")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Directory to save ALS model")
    parser.add_argument("--min-ratings", type=int, default=20, help="Minimum ratings per user")
    parser.add_argument("--test-per-user", type=int, default=2, help="Holdout ratings per user")
    parser.add_argument("--rank", type=int, default=20, help="ALS rank")
    parser.add_argument("--max-iter", type=int, default=10, help="ALS max iterations")
    parser.add_argument("--reg-param", type=float, default=0.08, help="ALS regularization")
    parser.add_argument("--top-n", type=int, default=10, help="Top-N recommendations per user")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-ratings", type=int, default=0, help="Optional cap on ratings rows")
    parser.add_argument("--driver-mem", default="4g", help="Spark driver memory")
    parser.add_argument("--executor-mem", default="4g", help="Spark executor memory")
    parser.add_argument("--output-local-file", default="", help="Write recs to a single local CSV file")

    args = parser.parse_args()

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    spark = build_spark("MovieRecommenderALS", args.driver_mem, args.executor_mem)
    spark.sparkContext.setLogLevel("WARN")

    ratings_path = Path(args.ratings)
    movies_path = Path(args.movies)
    output_dir = Path(args.output)
    model_dir = Path(args.model_dir)

    max_ratings = args.max_ratings if args.max_ratings > 0 else None
    ratings = load_ratings(spark, ratings_path, max_ratings)
    movies = load_movies(spark, movies_path)

    ratings = filter_active_users(ratings, min_ratings=args.min_ratings)
    train, test = split_train_test(ratings, test_per_user=args.test_per_user, seed=args.seed)

    model = train_als(train, rank=args.rank, max_iter=args.max_iter, reg_param=args.reg_param)

    rmse, precision, recall, ndcg = evaluate_model(model, test, k=args.top_n)

    print(f"RMSE: {rmse:.4f}")
    print(f"Precision@{args.top_n}: {precision:.4f}")
    print(f"Recall@{args.top_n}: {recall:.4f}")
    print(f"NDCG@{args.top_n}: {ndcg:.4f}")

    if args.output_local_file:
        export_recommendations_local(model, movies, Path(args.output_local_file), top_n=args.top_n)
    else:
        export_recommendations(model, movies, output_dir, top_n=args.top_n)
    model.write().overwrite().save(str(model_dir))

    spark.stop()


if __name__ == "__main__":
    main()
