import argparse #used to take input from command line
import os  # interaction with OS.
import sys  #python functions and variables. 
from pathlib import  Path  # for handling file paths in a platform-independent way.

import pandas as pd
from pyspark.sql import SparkSession, Window  #imports the Spark engine interface.

#built-in spark operations (functions) that you can use on Spark DataFrames.(to work with data)
from pyspark.sql.functions import ( 
    expr,
    size,
    array_intersect,
    avg,
    explode
)
from pyspark.ml.recommendation import ALS  #imports the ALS algorithm for collaborative filtering.
from pyspark.ml.evaluation import RegressionEvaluator # used to measure accuracy(RMSE).


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Go 2 folders up from this file That’s your project root

#these are default file paths of csv files , if user does not provide it then program should know where to find.

DEFAULT_RATINGS = PROJECT_ROOT / "data" / "raw" / "ratings.csv" # stroing path of ratings.csv file in a variable.
DEFAULT_MOVIES = PROJECT_ROOT / "data" / "raw" / "movies.csv" # stroing path of movies.csv file in a variable.
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "als_recommendations"  # stroing path of output directory for ALS recommendations in a variable.
DEFAULT_MODEL_DIR = PROJECT_ROOT / "saved_models" / "als"

# Start the spark engine
# A function that creates and returns a SparkSession. Takes 3 inputs: a name for the app, 
# memory for the driver (your laptop/master), and memory for executors (workers).
def build_spark(app_name: str, driver_mem: str, executor_mem: str) -> SparkSession:
    return (
        SparkSession.builder     # creating new spark session.
        .appName(app_name)  #Gives your Spark job a name — shows up in the Spark UI for monitoring.
        .config("spark.driver.memory", driver_mem)       #Sets how much RAM the driver and each worker gets
        .config("spark.executor.memory", executor_mem)   #More memory = can handle bigger data without crashing.
        .config("spark.sql.shuffle.partitions", "50")    #  Partitions = how many chunks Spark splits data into.
        .config("spark.default.parallelism", "50")       #  This controls how parallel the work is.
        .getOrCreate()   #Either creates a new session or reuses an existing one if it's already running.
                         # Prevents creating duplicate sessions.
    )

# Takes the SparkSession(to access spark operations), the file path, and an optional row limit.
def load_ratings(spark: SparkSession, path: Path, max_ratings: int | None):
    ratings = (
        spark.read                    # gives us a spark tool(dataframe reader object) to read data.
        .option("header", True)       #first row of CSV is column names, not data
        .option("inferSchema", True)  #Spark guesses the data types (int, float, string) automatically
        .csv(str(path))               #reads the file. str() converts Path object to a string.
    )

    #Validation check:If the CSV is missing any of the 3 required columns, crash immediately with a clear error message.
    required = {"userId", "movieId", "rating"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"ratings.csv missing columns: {sorted(missing)}")
    

    # .select() = picks only the columns you need (drops extras)
    # .cast("int")= type casting to int.
    # alias = rename the col after casting.

    ratings = ratings.select(
        col("userId").cast("int").alias("userId"),
        col("movieId").cast("int").alias("movieId"),
        col("rating").cast("double").alias("rating"),
        *([col("timestamp").cast("long")] if "timestamp" in ratings.columns else [])
    )

    if max_ratings:
        ratings = ratings.orderBy(expr("rand()"))  # sort by random numbers.
        ratings = ratings.limit(max_ratings)     # take only first N rows after shuffling.

    return ratings     

# load movies data.
def load_movies(spark: SparkSession, path: Path):
    movies = (
        spark.read                  # spark tool(dataframe reader object) to read data.
        .option("header", True)      # first row of CSV is column names, not data
        .option("inferSchema", True) # Spark guesses the data types (int, float, string) automatically
        .csv(str(path))              # reads the file. str() converts Path object to a string.
    )

    required = {"movieId", "title", "genres"}
    missing = required - set(movies.columns)
    if missing:
        raise ValueError(f"movies.csv missing columns: {sorted(missing)}")
    
    # select only required cols and cast it in "int" and do renaming of cols.
    return movies.select(
        col("movieId").cast("int").alias("movieId"),  
        col("title").cast("string").alias("title"),
        col("genres").cast("string").alias("genres")
    )

# Removes inactive user. Takes ratings data and a minimum threshold 
#(Users with too few ratings don't give the model enough signal to learn from.)
def filter_active_users(ratings, min_ratings: int):
    counts = ratings.groupBy("userId").agg(spark_count("rating").alias("cnt"))    #Groups by user and counts how many ratings each user gave
    active = counts.filter(col("cnt") >= min_ratings).select("userId")          #Filters to keep only users who have rated at least min_ratings movies. Then selects just the userId column.
    return ratings.join(active, on="userId", how="inner")      #Inner join — only keep ratings rows where the userId appears in the active list. This removes all ratings from users who didn't meet the threshold.


#Split Data for Training & Testing.
# test_per_user = how many ratings to test user.
# ratings = full dataset.
def split_train_test(ratings, test_per_user: int, seed: int):
    if "timestamp" in ratings.columns:
        window = Window.partitionBy("userId").orderBy(col("timestamp").desc()) #create a window per user and sort ratings.
        ranked = ratings.withColumn("rn", row_number().over(window))  #assign rank 1 to most recent rating, 2 to second most.
        test = ranked.filter(col("rn") <= test_per_user).drop("rn")
        train = ranked.filter(col("rn") > test_per_user).drop("rn")

    #If no timestamps, do a random 80/20 split.
    else:
        train, test = ratings.randomSplit([0.8, 0.2], seed)
        train_users = train.select("userId").distinct()
        test = test.join(train_users, on="userId", how="inner")

    return train, test

# Train the Recommender Model.
# train = training dataset (user-movie ratings)
# rank = number of latent factors.
# max_iter = training iteration in weight updates.
# reg_param = regularization parameter to prevent overfitting.
def train_als(train, rank: int, max_iter: int, reg_param: float):
    # als=creating instance of ALS algo.
    als = ALS(
        userCol="userId",           # defining cols like which is userid, movieid for ALS to understand.
        itemCol="movieId",
        ratingCol="rating",
        nonnegative=True,           # Forces all learned values to be ≥ 0.(i.e rating should be 1 to 5)
        coldStartStrategy="drop",   # this parameter is a part of model configurationif a user/movie in test wasn't in training, drop those rows instead of predicting NaN
        implicitPrefs=False,        # we are using explicit rating so implicit = false.
        rank=rank,                  # number of latent factors.
        maxIter=max_iter,
        regParam=reg_param
    )
    return als.fit(train)   # actual training of ALS on training data.

# Applying RMSE to evaluate model using Precision , Recall and NDCG for top-K recommendations.
def evaluate_model(model, test, k: int):

    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    predictions = model.transform(test)    # model predicts ratings for the test set
    rmse = evaluator.evaluate(predictions) # compares predicted vs actual ratings, returns RMSE

    # For each user, collect the set of movies they actually rated in the test set.
    actual = test.groupBy("userId").agg(expr("collect_set(movieId) as actual"))

    # Get top-K recommendations for every user, then extract just the movieIds from the recommendation structs.
    recs = model.recommendForAllUsers(k)
    recs = recs.select(
        "userId",
        expr("transform(recommendations, x -> x.movieId) as recs")
    )

    #Join the actual movies watched with the recommended movies per user.
    joined = actual.join(recs, on="userId", how="inner")

    # CalculatingNDCG for each user.
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



# Model = ALS trained model ready for making recommendations.
# Movies dataset (DataFrame) with features movieId, title, genres.
# output_dir = where to save the recommendations (as CSV files).
# top_n = how many recommendations per user to generate.

# This function generates top-N recommendations for each user using the trained ALS model,  then,
# joins them with movie metadata, and exports the results as CSV files in a distributed manner(spark based export so multiple csv files.

def export_recommendations(model, movies, output_dir: Path, top_n: int):
    recs = model.recommendForAllUsers(top_n)  # generates recommendations.

    # explode converts array of recommendations into separate rows.
    exploded = recs.select(
        col("userId"),
        explode(col("recommendations")).alias("rec")
    ).select(
        col("userId"),
        col("rec.movieId").alias("movieId"),
        col("rec.rating").alias("score")
    )


    # join add movie details (title, genres) to each recommended movie by matching on movieId.
    enriched = exploded.join(movies, on="movieId", how="left") 

    (
        enriched
        .orderBy(col("userId"), col("score").desc())
        .write
        .mode("overwrite")
        .option("header", True)
        .csv(str(output_dir))
    )

#Converts Spark data → Pandas DataFrame and Output is single CSV file

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
    parser.add_argument("--seed", type=int, default=42, help="Random seed") #calling 
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
