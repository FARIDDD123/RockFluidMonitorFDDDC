from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col

spark = SparkSession.builder \
    .appName("WriteToPostgres") \
    .config("spark.jars", "/home/mahdis/project/jars/postgresql-42.5.4.jar") \
    .getOrCreate()

schema = "WOB DOUBLE, RPM DOUBLE, Torque DOUBLE, ROP DOUBLE, BitArea DOUBLE, MSE DOUBLE"

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "enriched_features") \
    .load()

df_parsed = df.selectExpr("CAST(value AS STRING) as json_str")
df_json = df_parsed.select(from_json(col("json_str"), schema).alias("data")).select("data.*")

jdbc_url = "jdbc:postgresql://localhost:5432/drilling"
properties = {"user": "postgres", "password": "your_password", "driver": "org.postgresql.Driver"}

query = df_json.writeStream \
    .foreachBatch(lambda batch_df, _: batch_df.write.jdbc(jdbc_url, "enriched_features", mode="append", properties=properties)) \
    .option("checkpointLocation", "/tmp/checkpoints/postgres") \
    .start()

query.awaitTermination()
