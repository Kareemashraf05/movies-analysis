"""Render the headline KPIs as a grid of dashboard-style cards (PNG)."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Accent color per card -- edit here to restyle the whole dashboard
CARD_COLORS = [
    "#2E5EAA", "#2E5EAA", "#2E5EAA", "#8E44AD",
    "#27AE60", "#27AE60", "#D68910", "#C0392B",
]


def render_kpi_cards(kpis: dict, output_path: Path) -> None:
    """Draw the 8 headline KPIs as cards and save to output_path (PNG)."""
    cards = [
        ("TOTAL MOVIES", f"{kpis['total_movies']:,}",
         f"{kpis['year_min']}–{kpis['year_max']}"),
        ("AVG RATING", f"{kpis['avg_rating']:.2f}/10",
         f"{kpis['avg_vote_count']:.0f} avg votes"),
        ("AVG RUNTIME", f"{kpis['avg_runtime']:.0f} min", ""),
        ("TOP GENRE", kpis["top_genre"],
         f"{kpis['top_genre_count']:,} movies"),
        ("AVG BUDGET", f"${kpis['avg_budget']/1e6:.1f}M",
         f"{kpis['pct_budget_known']:.0f}% of movies have data"),
        ("AVG REVENUE", f"${kpis['avg_revenue']/1e6:.1f}M",
         f"{kpis['pct_revenue_known']:.0f}% of movies have data"),
        ("MEDIAN ROI", f"{kpis['median_roi']:.2f}x", "revenue / budget"),
        ("% PROFITABLE", f"{kpis['pct_profitable']:.1f}%", "revenue > budget"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 6))

    for ax, (label, value, sub), color in zip(axes.flat, cards, CARD_COLORS):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Card background
        ax.add_patch(FancyBboxPatch(
            (0.03, 0.05), 0.94, 0.9, transform=ax.transAxes,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            linewidth=0, facecolor=color, alpha=0.12))
        # Accent bar on the left edge
        ax.add_patch(FancyBboxPatch(
            (0.03, 0.05), 0.05, 0.9, transform=ax.transAxes,
            boxstyle="round,pad=0.0,rounding_size=0.025",
            linewidth=0, facecolor=color))

        ax.text(0.16, 0.60, value, transform=ax.transAxes, fontsize=21,
                fontweight="bold", color="#1a1a1a", va="center")
        ax.text(0.16, 0.30, label, transform=ax.transAxes, fontsize=10,
                color="#555555", va="center", fontweight="bold")
        if sub:
            ax.text(0.16, 0.14, sub, transform=ax.transAxes, fontsize=8.5,
                    color="#888888", va="center")

    fig.suptitle("Movies Dataset — Key KPIs", fontsize=17, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
