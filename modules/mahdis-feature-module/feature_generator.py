from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr

spark = SparkSession.builder \
    .appName("RealTimeFeatureGenerator") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:3.4.2") \
    .getOrCreate()

schema = "WOB DOUBLE, RPM DOUBLE, Torque DOUBLE, ROP DOUBLE, BitArea DOUBLE"

df_raw = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "raw_sensors") \
    .load()

df_json = df_raw.selectExpr("CAST(value AS STRING) as json_str") \
    .selectExpr(f"from_json(json_str, '{schema}') as data") \
    .select("data.*")

df_features = df_json.withColumn(
    "MSE",
    (col("WOB") / col("BitArea")) + (120 * 3.14 * col("RPM") * col("Torque")) / (col("ROP") * col("BitArea"))
)

query = df_features.selectExpr("to_json(struct(*)) AS value") \
    .writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "enriched_features") \
    .option("checkpointLocation", "/tmp/checkpoints/features") \
    .start()

query.awaitTermination()
