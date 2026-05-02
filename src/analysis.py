
from utils.spark_session import get_spark
from pyspark.sql.functions import sum

def run_analysis():
    print("Running analysis...")

    spark = get_spark()

    df = spark.read.csv("data/silver/data.csv", header=True, inferSchema=True)

    gold = df.groupBy("user_id") \
        .agg(sum("total").alias("user_revenue"))

    gold.show()

    gold.write.mode("overwrite").parquet("data/gold/")

    print("Analysis completed")