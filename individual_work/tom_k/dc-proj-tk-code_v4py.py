import pyspark
from pathlib import Path
import json
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates


def overview_cds_and_vinyl():

    def to_year_rating(rec):
        """
        Map each review dict -> (year, (rating_sum, count))
        """
        ts = rec.get("timestamp")
        rating = rec.get("rating")

        if ts is None or rating is None:
            return None

        dt = datetime.utcfromtimestamp(ts / 1000.0)  # ms -> seconds
        year = dt.year

        return (year, (float(rating), 1))


    def save_year_stats(year_results, output_path: str):
        """
        Pure RDD version: write CSV via saveAsTextFile, no pyspark.sql.
        year_results is a list of (year, (avg_rating, count)).
        """
        sc = pyspark.SparkContext.getOrCreate()

        # Prepare header + data lines
        header = "year,avg_rating,num_ratings"
        data_lines = [
            f"{int(year)},{float(stats[0])},{int(stats[1])}"
            for year, stats in year_results
        ]

        # Create an RDD with header + lines
        rdd = sc.parallelize([header] + data_lines)

        # One output file, plain text CSV
        (
            rdd.coalesce(1)
            .saveAsTextFile(output_path)
        )

    print("\nStarting overview for CDs and Vinyl category...\n")
    category = "CDs_and_Vinyl"
    data_path = f'gs://msds-694-cohort-14-group9/data/{category}.jsonl'
    output_path = f'gs://msds-694-cohort-14-group9/output/{category}_yearly_avg_ratings/'
    sc = pyspark.SparkContext(appName="CDsAndVinylRDD").getOrCreate()
    sc.setLogLevel("ERROR")

    path = str(data_path)

    raw_rdd = sc.textFile(path)
    records_rdd = raw_rdd.map(json.loads)

    print("Number of records:", records_rdd.count())
    print("First record:", records_rdd.first())


    # (1) Map to (year, (rating, 1)) and drop malformed rows
    year_rating_pairs = (
        records_rdd
        .map(to_year_rating)
        .filter(lambda x: x is not None)
    )

    # (2) Aggregate sum of ratings and count per year
    #     (year, (sum_ratings, count))
    year_agg = year_rating_pairs.reduceByKey(
        lambda a, b: (a[0] + b[0], a[1] + b[1])
    )

    # (3) Compute (avg_rating, count) per year
    #     (year, (avg_rating, count))
    year_stats = year_agg.mapValues(lambda s: (s[0] / s[1], s[1]))

    # (4) Sort by year and collect
    year_results = year_stats.sortByKey().collect()

    # Quick check
    for year, (avg, cnt) in year_results:
        print(f"{year}: Average Rating: {avg:.3f}, Review Count: {cnt}")

    save_year_stats(year_results, output_path)

    sc.stop()
    print("\nEnding overview for CDs and Vinyl category.\n")


if __name__ == "__main__":
    overview_cds_and_vinyl()