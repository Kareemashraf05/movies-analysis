"""Entry point for the movies dataset KPI pipeline.

Usage:
    python main.py --csv data/movies_metadata.csv --out outputs
"""
import argparse
from pathlib import Path

from src.load_data import load_and_clean
from src.kpis import compute_kpis
from src.visualize import (
    plot_top_genres,
    plot_movies_per_year,
    plot_rating_distribution,
    plot_budget_vs_revenue,
    plot_correlation_heatmap,
)
from src.cards import render_kpi_cards


def main(csv_path: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_and_clean(csv_path)
    kpis = compute_kpis(df)

    render_kpi_cards(kpis, out / "kpi_cards.png")
    plot_top_genres(df, out / "top_genres.png")
    plot_movies_per_year(df, out / "movies_per_year.png")
    plot_rating_distribution(df, out / "rating_distribution.png")
    plot_budget_vs_revenue(df, out / "budget_vs_revenue.png")
    plot_correlation_heatmap(df, out / "correlation_heatmap.png")

    print(f"Total movies:     {kpis['total_movies']:,}")
    print(f"Avg rating:       {kpis['avg_rating']:.2f}/10")
    print(f"Avg budget:       ${kpis['avg_budget']:,.0f}")
    print(f"Avg revenue:      ${kpis['avg_revenue']:,.0f}")
    print(f"Median ROI:       {kpis['median_roi']:.2f}x")
    print(f"Top genre:        {kpis['top_genre']} ({kpis['top_genre_count']:,})")
    print(f"\nOutputs saved to {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Movies dataset KPI pipeline")
    parser.add_argument("--csv", default="data/movies_metadata.csv",
                         help="Path to movies_metadata.csv")
    parser.add_argument("--out", default="outputs",
                         help="Directory to write charts/cards to")
    args = parser.parse_args()
    main(args.csv, args.out)
