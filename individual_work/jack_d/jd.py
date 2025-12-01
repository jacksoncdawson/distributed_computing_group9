from pyspark import SparkContext
import json
import math
from collections import defaultdict


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
    vg_rdd = sc.textFile("../../data_samples/Video_Games_SAMPLE.jsonl").map(json.loads)
    b_rdd = sc.textFile("../../data_samples/Books_SAMPLE.jsonl").map(json.loads)
    
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

    rating_by_category = defaultdict(lambda: {"book": 0, "video_game": 0})
    for (category, rating), count in rating_counts:
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


if __name__ == "__main__":
    analyze_books_vs_videogames()
