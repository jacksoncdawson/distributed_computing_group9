from pyspark import SparkContext
import json
import math
from datetime import datetime

# change when ready to push to GCS
books_path = "distributed_computing_group9/data_samples/Books_SAMPLE.jsonl"
video_games_path = "distributed_computing_group9/data_samples/Video_Games_SAMPLE.jsonl"


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


# Define Paul's Function
from pyspark import SparkConf, SparkContext
import json

def process_kindle_reviews_full():
    sc = SparkContext()
    
    rdd = sc.textFile("gs://msds-694-cohort-14-group9/data/Kindle_Store.jsonl")

    parsed_rdd = rdd.map(lambda line: json.loads(line))

    sample_rdd = parsed_rdd.takeSample(False, 5000, seed=42)
    sample_rdd = sc.parallelize(sample_rdd)

    sample_rdd = sample_rdd.map(
        lambda x: (
            str(x.get('asin', '')),
            str(x.get('parent_asin', '')),
            str(x.get('title', '')),
            str(x.get('text', '')),
            float(x.get('rating', 0.0)),
            int(x.get('timestamp', 0)),
            str(x.get('user_id', '')),
            bool(x.get('verified_purchase', False)),
            int(x.get('helpful_vote', 0)),
            int(len(x.get('images', [])))
        )
    )

    # Filter for verified reviews

    valid_reviews = sample_rdd.filter(lambda x: x[7] == True)

    # Print sample of RDDs
    print("Sample RDD (first 5 rows):")
    print(sample_rdd.take(5))

    print("Valid (verified purchase) reviews (first 5 rows):")
    print(valid_reviews.take(5))

    # STATISTICS ON RATINGS

    ratings = valid_reviews.map(lambda x: x[4])

    count = ratings.count()
    mean = ratings.mean()
    stdev = ratings.stdev()
    min_val = ratings.min()
    max_val = ratings.max()

    print("\nRating Summary:")
    print((count, mean, stdev, min_val, max_val))

    # AVG RATING BY # OF HELPFUL VOTES
    pair_rdd = valid_reviews.map(lambda x: (x[8], (x[4], 1)))

    reduced = pair_rdd.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))

    avg_ratings = reduced.mapValues(lambda x: (x[0] / x[1], x[1]))

    result = avg_ratings.sortByKey()

    print("\n(# helpful votes → (avg rating, # reviews)):")
    for row in result.take(40):
        print(row)

    # GOOD vs NOT-GOOD (>=4 stars)
    good = valid_reviews.map(lambda x: (1 if x[4] >= 4 else 0, 1))
    counts_good = good.reduceByKey(lambda x, y: x + y)

    print("\nGood review counts (>=4 stars):")
    print(counts_good.collect())

    # BAD vs NOT-BAD (<3 stars)
    bad = sample_rdd.map(lambda x: (1 if x[4] < 3 else 0, 1))
    counts_bad = bad.reduceByKey(lambda x, y: x + y)

    print("\nBad review counts (<3 stars):")
    print(counts_bad.collect())

    # AVG TEXT LENGTH OF VALID REVIEWS
    comb = valid_reviews.map(lambda x: (len(x[3]), 1))

    sum_length, count_length = comb.reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    avg_length = sum_length / count_length

    print("\nAverage length of review text (overall):")
    print(avg_length)

    # AVG TEXT LENGTH BY RATING
    pair_rdd_len = valid_reviews.map(lambda x: (x[4], (len(x[3]), 1)))

    reduced_len = pair_rdd_len.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))

    avgLen = reduced_len.mapValues(lambda x: x[0] / x[1])

    print("\nAverage review length by star rating:")
    for row in avgLen.sortByKey().collect():
        print(row)

    # Return everything 
    return {
        "sample_rdd": sample_rdd,
        "valid_reviews": valid_reviews,
        "rating_summary": (count, mean, stdev, min_val, max_val),
        "helpful_vote_stats": result.take(40),
        "good_counts": counts_good.collect(),
        "bad_counts": counts_bad.collect(),
        "avg_review_length": avg_length,
        "avg_length_by_rating": avgLen.collect()
    }


# Tom's Function
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
        rdd.coalesce(1).saveAsTextFile(output_path)
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


# --- RUN SCRIPTS ---

if __name__ == "__main__":
    
    # Run Jack's Script
    analyze_books_vs_videogames()
    # Run Paul's
    process_kindle_reviews_full()
    # Run Tom's
    overview_cds_and_vinyl()
    
    
