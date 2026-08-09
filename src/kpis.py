"""Compute headline KPIs from the cleaned movies DataFrame."""
import pandas as pd


def compute_kpis(df: pd.DataFrame) -> dict:
    """Return a dict of the KPI values used across the report and cards."""
    known_budget = df[df["budget_known"]]
    known_revenue = df[df["revenue_known"]]

    both_known = df[df["budget_known"] & df["revenue_known"]].copy()
    both_known["roi"] = both_known["revenue"] / both_known["budget"]

    genre_counts = df.explode("genre_list")["genre_list"].value_counts()
    genre_counts = genre_counts[genre_counts.index != ""]

    return {
        "total_movies": len(df),
        "year_min": int(df["release_year"].min()),
        "year_max": int(df["release_year"].max()),
        "avg_rating": df["vote_average"].mean(),
        "avg_runtime": df["runtime"].mean(),
        "avg_vote_count": df["vote_count"].mean(),
        "pct_budget_known": known_budget.shape[0] / len(df) * 100,
        "pct_revenue_known": known_revenue.shape[0] / len(df) * 100,
        "avg_budget": known_budget["budget"].mean(),
        "avg_revenue": known_revenue["revenue"].mean(),
        "median_roi": both_known["roi"].median(),
        "pct_profitable": (both_known["revenue"] > both_known["budget"]).mean() * 100,
        "total_revenue": both_known["revenue"].sum(),
        "total_budget": both_known["budget"].sum(),
        "top_genre": genre_counts.index[0],
        "top_genre_count": int(genre_counts.iloc[0]),
        "genre_counts": genre_counts,
    }
