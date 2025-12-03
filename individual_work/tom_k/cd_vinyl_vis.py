# cd_vinyl_vis.py

import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # Folder where this .py file lives
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ---- INPUT CSV PATH ----
    # Put your CSV in the same folder as this .py file and update the filename if needed.
    csv_filename = "output_CDs_and_Vinyl_yearly_avg_ratings_20251202_070943_part-00000"
    input_path = os.path.join(script_dir, csv_filename)

    # Read the CSV
    df = pd.read_csv(input_path)

    # Make sure it's sorted by year
    df = df.sort_values("year")

    years = df["year"]
    avg_ratings = df["avg_rating"]
    num_ratings = df["num_ratings"]

    # Total number of ratings across all years
    total_ratings = int(num_ratings.sum())

    # ---- PLOT ----
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Line plot for average rating (left y-axis)
    ax1.plot(years, avg_ratings, marker="o", linewidth=2)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Average Rating")

    # Bar plot for number of ratings (right y-axis)
    ax2 = ax1.twinx()
    ax2.bar(years, num_ratings, alpha=0.3)
    ax2.set_ylabel("Number of Ratings")

    # Title with total ratings
    title = (
        f"Average Rating and Number of Ratings per Year "
        f"(Total Ratings = {total_ratings:,})"
    )
    plt.title(title)

    fig.tight_layout()

    # ---- SAVE PLOT ----
    output_filename = "cd_vinyl_avg_rating_num_ratings.png"
    output_path = os.path.join(script_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()