from smart_open import open
import json
import pyspark
import nltk # pip install nltk
from nltk.corpus import stopwords

# For uploading data
bucket = "msds-694-cohort-14-group9/data"
filename = "Electronics.jsonl"
path = f"gs://{bucket}/{filename}"

first_5000 = []

with open(path, "r") as f:
    for i, line in enumerate(f):
        if i >= 5000:
            break
        first_5000.append(json.loads(line))

# RDD to be passed into each function
sc = pyspark.SparkContext()
base_rdd = sc.parallelize(first_5000)

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
