from pyspark import SparkContext, SparkConf
import json
import math
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

# for patrick's script
import nltk  # pip install nltk
from nltk.corpus import stopwords

# GCS Paths
books_path = "gs://msds-694-cohort-14-group9/data/Books.jsonl"
video_games_path = "gs://msds-694-cohort-14-group9/data/Video_Games.jsonl"
kindle_path = "gs://msds-694-cohort-14-group9/data/Kindle_Store.jsonl"

GCS_OUTPUT_BUCKET = "gs://msds-694-cohort-14-group9/output"

# Helper Function for Saving Visualizations to GCS
def save_chart_image_to_gcs(
    data, chart_type, title, x_label, y_label, filename, gcs_output_path
):
    """Generates a chart and saves the image file directly to a GCS path."""

    plt.figure(figsize=(10, 6))

    if chart_type == "hist":
        plt.hist(data, bins=np.arange(0.5, 6.5, 1.0), edgecolor="black", rwidth=0.8)
        plt.xticks(np.arange(1, 6, 1))

    elif chart_type == "bar_avg_rating":
        max_votes_to_plot = 20
        filtered_data = [
            item for item in data if item[0] <= max_votes_to_plot and item[2] > 0
        ]
        if not filtered_data:
            print(f"Skipping chart save for '{title}': No valid data.")
            plt.close()
            return

        keys = [item[0] for item in filtered_data]
        values = [item[1] for item in filtered_data]
        plt.bar(keys, values, color="skyblue")
        plt.xticks(keys)
        plt.ylim(0, 5.5)

    elif chart_type == "pie":
        good_count = next((c for c_type, c in data if c_type == 1), 0)
        not_good_count = next((c for c_type, c in data if c_type == 0), 0)

        labels = [
            f"Good (>=4 Stars) - {good_count}",
            f"Not Good (<4 Stars) - {not_good_count}",
        ]
        sizes = [good_count, not_good_count]

        plt.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=["#4CAF50", "#F44336"],
        )
        plt.axis("equal")

    elif chart_type == "scatter_len":
        keys = [item[0] for item in data]
        values = [item[1] for item in data]
        plt.scatter(keys, values, color="purple", s=100)
        plt.xticks(np.arange(1, 6, 1))

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(axis="y", linestyle="--")

    gcs_filepath = os.path.join(gcs_output_path, filename)
    plt.savefig(gcs_filepath)
    plt.close()
    print(f"Successfully saved visualization to GCS: {gcs_filepath}")


# --- SCRIPT DEFINITIONS ---

# Define Jack's Script
def analyze_books_vs_videogames():
    """
    Analyze consumer sentiment differences between video-game products and book products.

    This function:
    - Loads Books and Video Games sample data
    - Merges the datasets with category labels
    - Computes side-by-side comparison statistics
    - Provides detailed breakdowns of ratings, review lengths, helpfulness, and verified purchases
    """



    # Setup SparkContext
    sc = SparkContext.getOrCreate()

    # Read JSONL files as text RDDs and parse each line
    vg_rdd = sc.textFile(video_games_path).map(json.loads)
    b_rdd = sc.textFile(books_path).map(json.loads)

    # Add category field to each record
    vg_rdd = vg_rdd.map(lambda row: {**row, "category": "video_game"})
    b_rdd = b_rdd.map(lambda row: {**row, "category": "book"})

    # Union the two RDDs
    rdd = vg_rdd.union(b_rdd)

    # Add text length and title length fields for analysis
    rdd = rdd.map(
        lambda row: {
            **row,
            "text_length": len(row["text"]) if row["text"] else 0,
            "title_length": len(row["title"]) if row["title"] else 0,
        }
    )

    # ---- Side-by-Side Comparison Statistics ----

    # 1. Map each record into (category, value_dict)
    pair_rdd = rdd.map(
        lambda row: (
            row["category"],
            {
                "count": 1,
                "rating_sum": row["rating"],
                "rating_sq_sum": row["rating"] ** 2,
                "text_sum": row["text_length"],
                "text_sq_sum": row["text_length"] ** 2,
                "title_sum": row["title_length"],
                "helpful_sum": row["helpful_vote"],
                "verified_sum": 1 if row["verified_purchase"] else 0,
                "image_count_sum": len(row["images"]),
            },
        )
    )

    # 2. Combine values by category
    def combiner(v):
        return v

    def merger(a, b):
        return {
            "count": a["count"] + b["count"],
            "rating_sum": a["rating_sum"] + b["rating_sum"],
            "rating_sq_sum": a["rating_sq_sum"] + b["rating_sq_sum"],
            "text_sum": a["text_sum"] + b["text_sum"],
            "text_sq_sum": a["text_sq_sum"] + b["text_sq_sum"],
            "title_sum": a["title_sum"] + b["title_sum"],
            "helpful_sum": a["helpful_sum"] + b["helpful_sum"],
            "verified_sum": a["verified_sum"] + b["verified_sum"],
            "image_count_sum": a["image_count_sum"] + b["image_count_sum"],
        }

    agg = pair_rdd.combineByKey(combiner, merger, merger)

    # 3. Compute final statistics
    def finalize(category, m):
        n = m["count"]
        return {
            "category": category,
            "total_reviews": n,
            "avg_rating": m["rating_sum"] / n,
            "stddev_rating": math.sqrt(
                (m["rating_sq_sum"] / n) - (m["rating_sum"] / n) ** 2
            ),
            "avg_review_length": m["text_sum"] / n,
            "stddev_review_length": math.sqrt(
                (m["text_sq_sum"] / n) - (m["text_sum"] / n) ** 2
            ),
            "avg_title_length": m["title_sum"] / n,
            "avg_helpful_votes": m["helpful_sum"] / n,
            "total_helpful_votes": m["helpful_sum"],
            "verified_purchases": m["verified_sum"],
            "avg_images_per_review": m["image_count_sum"] / n,
        }

    final_rdd = agg.map(lambda kv: finalize(kv[0], kv[1]))

    # 4. Display results from RDD
    results = final_rdd.sortBy(lambda row: row["category"]).collect()
    for result in results:
        print(f"\nCategory: {result['category']}")
        print(f"  Total Reviews: {result['total_reviews']}")
        print(f"  Avg Rating: {result['avg_rating']:.2f}")
        print(f"  Stddev Rating: {result['stddev_rating']:.2f}")
        print(f"  Avg Review Length: {result['avg_review_length']:.2f}")
        print(f"  Stddev Review Length: {result['stddev_review_length']:.2f}")
        print(f"  Avg Title Length: {result['avg_title_length']:.2f}")
        print(f"  Avg Helpful Votes: {result['avg_helpful_votes']:.2f}")
        print(f"  Total Helpful Votes: {result['total_helpful_votes']}")
        print(f"  Verified Purchases: {result['verified_purchases']}")
        print(f"  Avg Images Per Review: {result['avg_images_per_review']:.2f}")

    # ---- Additional Detailed Breakdowns ----

    # Rating Distribution Comparison
    print("\nRATING DISTRIBUTION COMPARISON")
    print("=" * 80)
    rating_counts = (
        rdd.map(lambda row: ((row["category"], row["rating"]), 1))
        .reduceByKey(lambda a, b: a + b)
        .collect()
    )

    rating_by_category = {}
    for (category, rating), count in rating_counts:
        if rating not in rating_by_category:
            rating_by_category[rating] = {"book": 0, "video_game": 0}
        rating_by_category[rating][category] = count

    print(f"{'rating':<10} {'book':<15} {'video_game':<15}")
    print("-" * 40)
    for rating in sorted(rating_by_category.keys()):
        counts = rating_by_category[rating]
        print(f"{rating:<10} {counts['book']:<15} {counts['video_game']:<15}")

    # Review Length Quartiles
    print("\n\nREVIEW LENGTH QUARTILES")
    print("=" * 80)
    category_lengths = (
        rdd.map(lambda row: (row["category"], row["text_length"]))
        .groupByKey()
        .mapValues(list)
        .collect()
    )

    for category, lengths in sorted(category_lengths):
        sorted_lengths = sorted(lengths)
        n = len(sorted_lengths)
        q1_idx = n // 4
        q2_idx = n // 2
        q3_idx = 3 * n // 4

        print(f"\nCategory: {category}")
        print(f"  Min Length: {min(sorted_lengths)}")
        print(f"  25th Percentile: {sorted_lengths[q1_idx]}")
        print(f"  Median: {sorted_lengths[q2_idx]}")
        print(f"  75th Percentile: {sorted_lengths[q3_idx]}")
        print(f"  Max Length: {max(sorted_lengths)}")

    # Helpfulness Statistics
    print("\n\nHELPFULNESS STATISTICS")
    print("=" * 80)
    helpful_stats = (
        rdd.map(
            lambda row: (
                row["category"],
                {
                    "max_votes": row["helpful_vote"],
                    "has_votes": 1 if row["helpful_vote"] > 0 else 0,
                },
            )
        )
        .reduceByKey(
            lambda a, b: {
                "max_votes": max(a["max_votes"], b["max_votes"]),
                "has_votes": a["has_votes"] + b["has_votes"],
            }
        )
        .collect()
    )

    for category, stats in sorted(helpful_stats):
        print(f"\nCategory: {category}")
        print(f"  Max Helpful Votes: {stats['max_votes']}")
        print(f"  Reviews with Helpful Votes: {stats['has_votes']}")

    # Verified Purchase Analysis
    print("\n\nVERIFIED PURCHASE ANALYSIS")
    print("=" * 80)
    verification = (
        rdd.map(
            lambda row: (
                row["category"],
                {"total": 1, "verified": 1 if row["verified_purchase"] else 0},
            )
        )
        .reduceByKey(
            lambda a, b: {
                "total": a["total"] + b["total"],
                "verified": a["verified"] + b["verified"],
            }
        )
        .collect()
    )

    for category, stats in sorted(verification):
        percentage = (stats["verified"] / stats["total"]) * 100
        print(f"\nCategory: {category}")
        print(f"  Total Reviews: {stats['total']}")
        print(f"  Verified Count: {stats['verified']}")
        print(f"  Verified Percentage: {percentage:.2f}%")

    sc.stop()


# Define Paul's Script
def process_kindle_reviews_full():
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    
    print("Starting PySpark analysis on Dataproc...")
    
    # 1. READ from GCS input path
    rdd = sc.textFile(kindle_path) 
    parsed_rdd = rdd.map(lambda line: json.loads(line))
    
    # Sampling and structuring RDDs (Logic remains the same)
    sample_rdd = parsed_rdd.takeSample(False, 5000, seed=42)
    sample_rdd = sc.parallelize(sample_rdd)

    sample_rdd = sample_rdd.map(
        lambda x: (
            str(x.get('asin', '')), str(x.get('parent_asin', '')), str(x.get('title', '')),
            str(x.get('text', '')), float(x.get('rating', 0.0)), int(x.get('timestamp', 0)),
            str(x.get('user_id', '')), bool(x.get('verified_purchase', False)), 
            int(x.get('helpful_vote', 0)), int(len(x.get('images', [])))
        )
    )

    valid_reviews = sample_rdd.filter(lambda x: x[7] == True).cache()
    
    # RATING DISTRIBUTION (Histogram)
    print("\n--- 1. Saving Rating Distribution Histogram to GCS ---")
    ratings = valid_reviews.map(lambda x: x[4]).cache() 
    rating_list = ratings.collect()
    
    save_chart_image_to_gcs(
        rating_list, 
        'hist', 
        'Distribution of Verified Review Ratings', 
        'Star Rating', 
        'Number of Reviews',
        '01_rating_distribution.png',
        GCS_OUTPUT_BUCKET
    )
    
    
    # AVG RATING BY HELPFUL VOTES (Bar Chart)
    pair_rdd = valid_reviews.map(lambda x: (x[8], (x[4], 1)))
    reduced = pair_rdd.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    avg_ratings_rdd = reduced.mapValues(lambda x: (x[0] / x[1], x[1])) 
    result = avg_ratings_rdd.sortByKey().map(lambda x: (x[0], x[1][0], x[1][1])).collect()
    
    save_chart_image_to_gcs(
        result, 
        'bar_avg_rating', 
        'Avg Rating by Helpful Vote Count (Up to 20 Votes)', 
        'Number of Helpful Votes', 
        'Average Star Rating',
        '02_avg_rating_by_votes.png',
        GCS_OUTPUT_BUCKET
    )
    
    # GOOD VS. NOT-GOOD RATIO (Pie Chart)
    good_vs_not = valid_reviews.map(lambda x: (1 if x[4] >= 4 else 0, 1))
    counts_good = good_vs_not.reduceByKey(lambda x, y: x + y).collect()
    
    save_chart_image_to_gcs(
        counts_good, 
        'pie', 
        'Verified Reviews: Good vs. Not Good (>=4 Stars)', 
        '', 
        '',
        '03_good_vs_not_good_ratio.png',
        GCS_OUTPUT_BUCKET
    )
    
    # AVG TEXT LENGTH BY RATING (Scatter Plot)
    pair_rdd_len = valid_reviews.map(lambda x: (x[4], (len(x[3]), 1)))
    reduced_len = pair_rdd_len.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    avgLen = reduced_len.mapValues(lambda x: x[0] / x[1])
    
    avg_length_by_rating_list = avgLen.sortByKey().collect()
    
    save_chart_image_to_gcs(
        avg_length_by_rating_list, 
        'scatter_len', 
        'Average Review Text Length by Star Rating', 
        'Star Rating', 
        'Average Text Length (Characters)',
        '04_avg_length_by_rating.png',
        GCS_OUTPUT_BUCKET
    )
    
    sc.stop()
    print(f"\nProcessing complete. All visualization images have been saved to the GCS path: {GCS_OUTPUT_BUCKET}")


# Define Tom's Script
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
        sc = SparkContext.getOrCreate()

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
        rdd.coalesce(1).saveAsTextFile(output_path)
        )

    print("\nStarting overview for CDs and Vinyl category...\n")
    category = "CDs_and_Vinyl"
    data_path = f'gs://msds-694-cohort-14-group9/data/{category}.jsonl'
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = f'gs://msds-694-cohort-14-group9/output/{category}_yearly_avg_ratings_{ts}/'
    sc = SparkContext(appName="CDsAndVinylRDD").getOrCreate()
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


# Define Patrick's Script
def analyze_electronics_reviews():

    # For uploading data
    bucket = "msds-694-cohort-14-group9/data"
    filename = "Electronics.jsonl"
    path = f"gs://{bucket}/{filename}"

    sc = SparkContext()
    base_rdd = sc.textFile(path).map(json.loads)

    def get_average_length_by_rating(base_rdd):

        average_review_length_by_rating = {}
        for i in range(5):
            rating = i+1
            filtered_rdd = base_rdd.filter(lambda x: int(x['rating']) == rating)
            review_lengths = filtered_rdd.map(lambda x: len(x['text']))
            average_review_length = review_lengths.reduce(lambda x, y: x+y)/filtered_rdd.count()

            average_review_length_by_rating[rating] = round(average_review_length, 2)

        return average_review_length_by_rating

    def get_top_words_by_rating(base_rdd):

        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

        def parse_review(text):
            problem_chars = ['.','><br','/', '-']
            filtered_text = text.lower()
            for c in problem_chars:
                filtered_text = filtered_text.replace(c,'')
            return filtered_text.split(' ')

        top_words_by_rating = {}
        for i in range(5):
            rating = i+1
            filtered_rdd = base_rdd.filter(lambda x: int(x['rating']) == rating)
            word_rdd = filtered_rdd.flatMap(lambda x: parse_review(x['text']))
            clean_word_rdd = word_rdd.filter(lambda x: x not in stop_words and x not in ['', '>'])

            n = clean_word_rdd.count()

            word_counts = clean_word_rdd.map(lambda word: (word, 1)).reduceByKey(lambda x, y: x + y)
            sorted_word_counts = word_counts.sortBy(lambda x: x[1], ascending=False)
            sorted_word_ratios = sorted_word_counts.mapValues(lambda x: round(x/n, 5))

            top_words_by_rating[rating] = sorted_word_ratios.take(10)

        return top_words_by_rating

    rating_lengths, top_words = get_average_length_by_rating(base_rdd), get_top_words_by_rating(base_rdd)
    print(rating_lengths)
    print(top_words)

    sc.stop()

    return rating_lengths, top_words


# Seb's Function
def analyze_ratings_from_file(file_path):
    """
    Full pipeline:
      1. Read newline-delimited JSON reviews from `file_path` into an RDD.
      2. Cast fields to proper types.
      3. Compute average rating & helpfulness per user.
      4. Calibrate user ratings to have mean 3.0 (for users with ≥ min_reviews_per_user).
      5. Compute per-ASIN original vs calibrated averages.
      6. Plot original vs calibrated product averages.

    Only argument required: file_path (string).
    """

    import json
    from datetime import datetime
    import matplotlib.pyplot as plt

    import pyspark
    from pyspark.sql import SparkSession

    sc = SparkContext.getOrCreate()
    spark = SparkSession.builder.getOrCreate()

    # Helper to create the typed RDD
    def createRDD(sc, file_path, field_order, types):
        raw_rdd = sc.textFile(file_path)

        def parse_json_line(line):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                return None

        def has_value(x):
            return x is not None and x != ""

        casters = {
            "int": lambda x: int(x) if has_value(x) else None,
            "float": lambda x: float(x) if has_value(x) else None,
            "str": lambda x: x if x is not None else "",
            "bool": lambda x: (
                x
                if isinstance(x, bool)
                else (
                    (x.lower() in ("1", "true", "t", "yes", "y"))
                    if has_value(x)
                    else None
                )
            ),
            "date": lambda x: (
                datetime.fromtimestamp(x / 1000.0).date() if has_value(x) else None
            ),
        }

        def cast_row_to_dict(record):
            casted_dict = {}
            for i, name in enumerate(field_order):
                value = record.get(name)
                target_type = types[i]
                casted_dict[name] = casters[target_type](value)
            return casted_dict

        return (
            raw_rdd.map(parse_json_line)
            .filter(lambda x: x is not None)
            .map(cast_row_to_dict)
        )

    # Average rating & helpfulness by user
    def avg_rate_by_user(rdd):
        # (user_id, (rating, helpful_vote))
        user_pairs_rdd = rdd.map(
            lambda x: (x["user_id"], (x["rating"], x["helpful_vote"]))
        )

        createCombiner = lambda value: (value[0], value[1], 1)
        mergeValue = lambda acc, val: (acc[0] + val[0], acc[1] + val[1], acc[2] + 1)
        mergeCombiners = lambda a1, a2: (a1[0] + a2[0], a1[1] + a2[1], a1[2] + a2[2])

        combined_totals_rdd = user_pairs_rdd.combineByKey(
            createCombiner, mergeValue, mergeCombiners
        )

        averages_rdd = combined_totals_rdd.map(
            lambda kv: {
                "user_id": kv[0],
                "avg_rating": round(kv[1][0] / kv[1][2], 2),
                "avg_helpfulness": round(kv[1][1] / kv[1][2], 2),
                "review_count": kv[1][2],
            }
        )

        # Sort by avg_helpfulness desc
        rdd_to_sort = averages_rdd.map(
            lambda record_dict: (record_dict["avg_helpfulness"], record_dict)
        )
        sorted_pair_rdd = rdd_to_sort.sortByKey(ascending=False)
        sorted_results_rdd = sorted_pair_rdd.map(lambda kv: kv[1])

        return sorted_results_rdd

    # Calibrated ratings per user
    def calibrated_ratings_per_user(
        avg_rating_per_user_rdd, ratings_rdd, min_reviews_per_user=3
    ):
        """
        Recalibrate ratings so each user's average rating becomes 3.0.
        Optionally ignore users with fewer than `min_reviews_per_user` reviews.
        """

        # (user_id, (avg_rating, review_count))
        avg_by_user_rdd = avg_rating_per_user_rdd.map(
            lambda d: (d["user_id"], (d["avg_rating"], d["review_count"]))
        )

        # (user_id, review_dict)
        ratings_by_user_rdd = ratings_rdd.map(lambda d: (d["user_id"], d))

        # (user_id, (review_dict, (avg_rating, review_count)))
        joined_rdd = ratings_by_user_rdd.join(avg_by_user_rdd)

        # use users with at least N reviews
        filtered_rdd = joined_rdd.filter(lambda kv: kv[1][1][1] >= min_reviews_per_user)

        def apply_calibration(kv):
            user_id, (review_dict, (avg_rating, review_count)) = kv

            shift = 3.0 - avg_rating
            new_review = review_dict.copy()
            raw_calibrated = new_review["rating"] + shift

            calibrated = max(1.0, min(5.0, raw_calibrated))
            new_review["calibrated_rating"] = round(calibrated, 2)

            return new_review

        calibrated_rdd = filtered_rdd.map(apply_calibration)
        return calibrated_rdd

    # Average rating by product helper
    def avg_rating_by_asin(rdd, rating_key):
        # (asin, (sum_rating, count))
        pair_rdd = rdd.map(lambda x: (x["asin"], (x[rating_key], 1)))
        summed_rdd = pair_rdd.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
        # (asin, (avg_rating, count))
        return summed_rdd.map(lambda kv: (kv[0], (kv[1][0] / kv[1][1], kv[1][1])))

    # Pipeline
    fields = [
        "rating",
        "title",
        "text",
        "images",
        "asin",
        "parent_asin",
        "user_id",
        "timestamp",
        "helpful_vote",
        "verified_purchase",
    ]
    types = ["int", "str", "str", "str", "str", "str", "str", "date", "int", "bool"]

    # Load and cast data
    ratings_rdd = createRDD(sc, file_path, fields, types)

    # Average rating per user
    avg_rating_per_user_rdd = avg_rate_by_user(ratings_rdd)

    # Calibrated ratings (using 3+ reviews as in your example)
    calib_rate_rdd = calibrated_ratings_per_user(
        avg_rating_per_user_rdd, ratings_rdd, min_reviews_per_user=3
    )

    # Original vs calibrated averages per product
    orig_avg_by_asin_rdd = avg_rating_by_asin(ratings_rdd, "rating")
    calib_avg_by_asin_rdd = avg_rating_by_asin(calib_rate_rdd, "calibrated_rating")

    # (asin, ((orig_avg, orig_count), (calib_avg, calib_count)))
    joined_rdd = orig_avg_by_asin_rdd.join(calib_avg_by_asin_rdd)

    # Keep only products with more than 10 ratings (using original count)
    filtered_rdd = joined_rdd.filter(lambda kv: kv[1][0][1] > 5)

    # Turn into dict RDD with the difference
    diff_rdd = filtered_rdd.map(
        lambda kv: {
            "asin": kv[0],
            "orig_avg": kv[1][0][0],
            "calib_avg": kv[1][1][0],
            "delta": kv[1][1][0] - kv[1][0][0],
            "count": kv[1][0][1],
        }
    )

    # Convert to DataFrame / Pandas and plot
    diff_df = spark.createDataFrame(diff_rdd)
    pdf = diff_df.toPandas()

    plt.figure(figsize=(6, 6))
    plt.scatter(pdf["orig_avg"], pdf["calib_avg"])
    plt.plot([1, 5], [1, 5])
    plt.xlabel("Original avg rating")
    plt.ylabel("Calibrated avg rating")
    plt.title("Product average: original vs calibrated")
    plt.tight_layout()
    plt.show()

    return {
        "ratings_rdd": ratings_rdd,
        "avg_rating_per_user_rdd": avg_rating_per_user_rdd,
        "calibrated_rdd": calib_rate_rdd,
        "diff_spark_df": diff_df,
        "diff_pandas_df": pdf,
    }


# --- RUN SCRIPTS ---

if __name__ == "__main__":
    
    # Run Jack's Script
    analyze_books_vs_videogames()
    # Run Paul's
    process_kindle_reviews_full()
    # Run Patrick's Script
    analyze_electronics_reviews()
    # Run Tom's
    overview_cds_and_vinyl()
    # Run Seb's
    analyze_ratings_from_file(video_games_path)
