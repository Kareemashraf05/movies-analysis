"""Chart generation for the movies KPI report."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def plot_top_genres(df: pd.DataFrame, output_path: Path, top_n: int = 10) -> None:
    genre_counts = df.explode("genre_list")["genre_list"].value_counts().head(top_n)
    genre_counts = genre_counts[genre_counts.index != ""]

    plt.figure(figsize=(10, 6))
    genre_counts.sort_values().plot(kind="barh", color="steelblue", edgecolor="black")
    plt.title(f"Top {top_n} Genres by Movie Count", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Movies")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_movies_per_year(df: pd.DataFrame, output_path: Path, since: int = 1930) -> None:
    yearly = df[df["release_year"] >= since].groupby("release_year").size()

    plt.figure(figsize=(12, 6))
    plt.plot(yearly.index, yearly.values, linewidth=2)
    plt.title(f"Movies Released Per Year ({since}–{int(df['release_year'].max())})",
              fontsize=14, fontweight="bold")
    plt.xlabel("Year")
    plt.ylabel("Number of Movies")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_rating_distribution(df: pd.DataFrame, output_path: Path, min_votes: int = 10) -> None:
    valid_votes = df[df["vote_count"] >= min_votes]["vote_average"]

    plt.figure(figsize=(10, 6))
    plt.hist(valid_votes, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
    plt.title(f"Rating Distribution (movies with {min_votes}+ votes)",
              fontsize=14, fontweight="bold")
    plt.xlabel("Vote Average (out of 10)")
    plt.ylabel("Number of Movies")
    plt.axvline(valid_votes.mean(), color="red", linestyle="--",
                label=f"Mean: {valid_votes.mean():.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_budget_vs_revenue(df: pd.DataFrame, output_path: Path) -> None:
    bv = df[(df["budget_known"]) & (df["revenue_known"])]

    plt.figure(figsize=(10, 6))
    plt.scatter(bv["budget"], bv["revenue"], alpha=0.4, s=15, color="steelblue")
    plt.plot([1e3, 1e9], [1e3, 1e9], color="red", linestyle="--", linewidth=1,
              label="Break-even line")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Budget vs Revenue (movies with both known, log-log scale)",
              fontsize=14, fontweight="bold")
    plt.xlabel("Budget ($, log scale)")
    plt.ylabel("Revenue ($, log scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    subset = df[df["budget_known"] & df["revenue_known"]]
    num_cols = subset[["budget", "revenue", "runtime", "vote_average",
                       "vote_count", "popularity"]]
    correlation = num_cols.corr()

    plt.figure(figsize=(8, 7))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", center=0, fmt=".2f",
                square=True, linewidths=0.5)
    plt.title("Correlation Matrix (movies w/ known budget & revenue)",
              fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
