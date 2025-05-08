import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, regexp_replace, split
from pyspark.sql.types import StringType, DoubleType

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: transform_traffy_data.py <input_gcs_path> <output_gcs_path>")
        sys.exit(-1)

    input_gcs_path = sys.argv[1]
    output_gcs_path = sys.argv[2]

    spark = SparkSession.builder.appName("TraffyDataTransform").getOrCreate()

    # Read data from GCS as CSV
    raw_df = spark.read.csv(
        input_gcs_path,
        header=True,
        inferSchema=True,
        multiLine=True,
        quote='"',
        escape='"',
        encoding="utf-8",
    )

    transformed_df = raw_df.select(
        col("ticket_id"),
        col("type"),
        col("organization"),
        col("comment"),
        col("photo"),
        col("photo_after"),
        col("coords"),
        col("address"),
        col("subdistrict"),
        col("district"),
        col("province"),
        to_timestamp(col("timestamp").cast(StringType())).alias("timestamp"),
        col("state"),
        col("star").cast("integer"),
        col("count_reopen").cast("integer"),
        to_timestamp(col("last_activity").cast(StringType())).alias("last_activity"),
    )

    # Clean string columns: replace unusual line terminators and newlines/carriage returns with spaces
    string_columns_to_clean = [
        "comment",
    ]

    # Create a list of select expressions for cleaning
    select_exprs = [
        (
            regexp_replace(
                regexp_replace(col(c), '"', ""), r"[\r\n\x85\u2028\u2029]+", " "
            ).alias(c)
            if c in string_columns_to_clean
            else col(c)
        )
        for c in transformed_df.columns
    ]

    transformed_df = transformed_df.select(*select_exprs)

    # Clean the 'coords' column and cast to DoubleType
    # Ensure coords is treated as a string before splitting, in case it was inferred differently
    transformed_df = transformed_df.withColumn(
        "coords_str", col("coords").cast(StringType())
    )

    split_coords = split(col("coords_str"), ",")
    transformed_df = transformed_df.withColumn(
        "longitude", split_coords.getItem(0).cast(DoubleType())
    ).withColumn("latitude", split_coords.getItem(1).cast(DoubleType()))

    # Select all columns except the original 'coords' and temporary 'coords_str'
    final_columns = [
        c for c in transformed_df.columns if c not in ["coords", "coords_str"]
    ]
    transformed_df = transformed_df.select(*final_columns)

    # Write transformed data to GCS in CSV format
    transformed_df.coalesce(1).write.csv(
        output_gcs_path, header=True, mode="overwrite", quoteAll=True
    )

    spark.stop()
