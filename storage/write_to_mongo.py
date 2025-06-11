from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("WriteToMongo") \
    .config("spark.mongodb.output.uri", "mongodb://localhost:27017/drilling.enriched_features") \
    .getOrCreate()

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "enriched_features") \
    .load()

df_parsed = df.selectExpr("CAST(value AS STRING) as json_str")  # برای داده json

# در اینجا بسته به فرمت داده، باید JSON رو به دیتافریم تبدیل کنی؛ می‌تونی با from_json تبدیل کنی.

# مثال:
from pyspark.sql.functions import from_json, col
schema = "WOB DOUBLE, RPM DOUBLE, Torque DOUBLE, ROP DOUBLE, BitArea DOUBLE, MSE DOUBLE"
df_json = df_parsed.select(from_json(col("json_str"), schema).alias("data")).select("data.*")

query = df_json.writeStream \
    .format("mongodb") \
    .option("checkpointLocation", "/tmp/checkpoints/mongo") \
    .start()

query.awaitTermination()
