"""Load and clean the movies_metadata.csv dataset."""
import ast

import pandas as pd


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV and apply standard cleaning steps.

    - Coerces budget/popularity/id to numeric (a handful of rows are corrupted
      in the source file and get dropped).
    - Parses release_date and derives release_year / decade.
    - Flags budget_known / revenue_known. In this dataset a 0 in either column
      means "not recorded", not a literal $0 movie -- treating it as real
      would silently crash every average.
    - Parses the stringified genres list (e.g. "[{'id': 16, 'name': 'Animation'}]")
      into a plain list of genre names.

    Returns the cleaned DataFrame.
    """
    df = pd.read_csv(csv_path, low_memory=False)

    for col in ("budget", "popularity", "id"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["id"])

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    df["decade"] = (df["release_year"] // 10 * 10)

    df["budget_known"] = df["budget"] > 0
    df["revenue_known"] = df["revenue"] > 0

    df["genre_list"] = df["genres"].apply(_parse_genres)

    return df


def _parse_genres(raw) -> list:
    try:
        return [g["name"] for g in ast.literal_eval(raw)]
    except (ValueError, SyntaxError, TypeError):
        return []
