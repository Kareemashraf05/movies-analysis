# -*- coding: utf-8 -*-
"""kpi_dashboard.py

Movies Dataset KPI Dashboard -- same structure as kpi1.py (Sales Dashboard),
adapted for The Movies Dataset (Kaggle, rounakbanik/the-movies-dataset).
Now includes an ML Prediction page, wired to train_model_project.py's model.pkl.
"""

# ==========================================
# Load Libraries
# ==========================================
import streamlit as st
import pandas as pd
import numpy as np
import ast
import os
import requests
import matplotlib.pyplot as plt

from joblib import load, dump
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(page_title="Movies Dashboard", layout="wide")

st.title("🎬 Movies Dashboard")
st.write("The Movies Dataset (TMDB / Kaggle)")


# ==========================================
# Load Dataset
# ==========================================

# Put "movies_metadata.csv" in a "data" folder next to kpi_dashboard.py --
# or don't, since it downloads automatically if missing (needed for cloud
# deployment, where the repo doesn't hold the 34MB CSV)

CSV_PATH = "data/movies_metadata.csv"
CSV_URL = "https://raw.githubusercontent.com/master-temp/movie-rec/main/movies_metadata.csv"


@st.cache_data
def load_dataset():
    if not os.path.exists(CSV_PATH):
        os.makedirs("data", exist_ok=True)
        response = requests.get(CSV_URL)
        response.raise_for_status()
        with open(CSV_PATH, "wb") as f:
            f.write(response.content)

    return pd.read_csv(CSV_PATH, low_memory=False)


df = load_dataset()

st.success("Dataset Loaded Successfully!")

st.subheader("Dataset Preview")

st.dataframe(df.head())

# ==========================================
# Dataset Preview
# ==========================================

print(df.head())

print("\nDataset Information:\n")
print(df.info())

print("\nMissing Values:\n")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# ==========================================
# Data Cleaning
# ==========================================

df.drop_duplicates(inplace=True)

for col in ["budget", "popularity", "id"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["id"])

df["release_date"] = pd.to_datetime(
    df["release_date"],
    format="mixed",
    errors="coerce"
)

df["release_year"] = df["release_date"].dt.year

# 0 in budget/revenue means "not recorded" in this dataset, not a real $0 movie
df["budget_known"] = df["budget"] > 0
df["revenue_known"] = df["revenue"] > 0


def parse_genres(raw):
    try:
        return [g["name"] for g in ast.literal_eval(raw)]
    except (ValueError, SyntaxError, TypeError):
        return []


df["genre_list"] = df["genres"].apply(parse_genres)

df["primary_genre"] = df["genre_list"].apply(
    lambda genres: genres[0] if len(genres) > 0 else "Unknown"
)

# belongs_to_collection is only populated for franchise/sequel titles
df["is_franchise"] = df["belongs_to_collection"].notna().astype(int)

print("\nCleaning Completed Successfully!")

print(df.head())


# ==========================================
# Load Trained Model
# ==========================================

# On cloud deployment there's no model.pkl in the repo (too big to upload),
# so train one on the fly the first time the app runs. @st.cache_resource
# means this only happens once per deployment, not on every interaction.

@st.cache_resource
def get_model(_training_df):
    if os.path.exists("model.pkl"):
        return load("model.pkl")

    model_df = _training_df[
        (_training_df["budget"] > 0) & (_training_df["revenue"] > 0)
    ].dropna(subset=["runtime", "primary_genre", "original_language", "release_year"])

    X = model_df[
        ["budget", "runtime", "release_year", "is_franchise", "primary_genre", "original_language"]
    ]
    y = (model_df["revenue"] > model_df["budget"]).astype(int)

    categorical_features = ["primary_genre", "original_language"]

    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
        remainder="passthrough"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    trained_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    trained_model.fit(X_train, y_train)

    try:
        dump(trained_model, "model.pkl")
    except OSError:
        pass  # read-only filesystem on some cloud environments -- fine, cache handles it

    return trained_model


model = get_model(df)


# ========== Page Selector ==========

page = st.sidebar.selectbox(
    "Select Page",
    [
        "Dashboard",

        "ML Prediction"
    ]
)


if page == "Dashboard":

    # ========== Filters ==========

    st.sidebar.header("Filters")

    # Genre Filter
    selected_genre = st.sidebar.selectbox(
        "Select Genre",
        ["All"] + sorted(df["primary_genre"].unique().tolist())
    )

    if selected_genre == "All":
        filtered_df = df.copy()
    else:
        filtered_df = df[df["primary_genre"] == selected_genre]

    # Only budget/revenue-known rows feed the money KPIs, same way the source
    # data treats 0 as "not recorded" rather than a literal $0 movie
    money_df = filtered_df[
        filtered_df["budget_known"] & filtered_df["revenue_known"]
    ]

    # ==========================================
    # KPI Monitoring
    # ==========================================

    # Total Revenue
    total_revenue = money_df["revenue"].sum()

    # Total Profit
    total_profit = (money_df["revenue"] - money_df["budget"]).sum()

    # Total Movies
    total_movies = filtered_df["id"].nunique()

    # Average Revenue
    average_revenue = money_df["revenue"].mean()

    # Average Rating
    average_rating = filtered_df["vote_average"].mean()

    # Average Runtime
    average_runtime = filtered_df["runtime"].mean()

    # Profit Margin
    profit_margin = (total_profit / total_revenue) * 100

    print("========== KPI Monitoring ==========\n")

    print(f"Total Revenue: ${total_revenue:,.2f}")
    print(f"Total Profit: ${total_profit:,.2f}")
    print(f"Total Movies: {total_movies}")
    print(f"Average Revenue: ${average_revenue:.2f}")
    print(f"Average Rating: {average_rating:.2f}")
    print(f"Average Runtime: {average_runtime:.2f} min")
    print(f"Profit Margin: {profit_margin:.2f}%")


    st.header("📈 KPI Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("💰 Total Revenue", f"${total_revenue:,.2f}")

    with col2:
        st.metric("💵 Total Profit", f"${total_profit:,.2f}")

    with col3:
        st.metric("🎬 Total Movies", total_movies)

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric("📊 Avg Revenue", f"${average_revenue:.2f}")

    with col5:
        st.metric("⭐ Avg Rating", f"{average_rating:.2f}/10")

    with col6:
        st.metric("⏱ Avg Runtime", f"{average_runtime:.2f} min")

    st.metric("📈 Profit Margin", f"{profit_margin:.2f}%")

    # ==========================================
    # Revenue by Genre
    # ==========================================

    revenue_by_genre = (
        money_df.groupby("primary_genre")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        revenue_by_genre.index,
        revenue_by_genre.values
    )

    ax.set_title("Revenue by Genre (Top 10)")
    ax.tick_params(axis="x", rotation=45)

    st.pyplot(fig)

    # ==========================================
    # Profit by Language
    # ==========================================

    top_languages = filtered_df["original_language"].value_counts().head(10).index

    profit_by_language = (
        money_df[money_df["original_language"].isin(top_languages)]
        .assign(profit=lambda d: d["revenue"] - d["budget"])
        .groupby("original_language")["profit"]
        .sum()
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        profit_by_language.index,
        profit_by_language.values
    )

    ax.set_title("Profit by Language (Top 10 Languages by Movie Count)")

    st.pyplot(fig)


    # ==========================================
    # Yearly Release Trend
    # ==========================================

    yearly_releases = filtered_df[
        filtered_df["release_year"] >= 1930
    ].groupby("release_year").size()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        yearly_releases.index,
        yearly_releases.values,
        marker="o"
    )

    ax.set_title("Movies Released Per Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Movies")

    st.pyplot(fig)


    # ==========================================
    # Top 10 Movies by Revenue
    # ==========================================

    top_movies = (
        money_df.groupby("title")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        top_movies.index,
        top_movies.values
    )

    ax.set_title("Top 10 Movies by Revenue")
    ax.set_xlabel("Revenue")
    ax.invert_yaxis()

    st.pyplot(fig)


    # ==========================================
    # Recommendation Logic
    # ==========================================

    st.header("💡 Business Recommendations")

    recommendations = []

    if profit_margin < 10:
        recommendations.append(
            "Review production budgets -- profit margin is thin relative to revenue."
        )

    if average_rating < 6:
        recommendations.append(
            "Average audience rating is below 6/10 -- prioritize script and production quality."
        )

    pct_budget_known = filtered_df["budget_known"].mean() * 100
    if pct_budget_known < 30:
        recommendations.append(
            "Budget/revenue reporting is missing for most titles -- improve data "
            "collection to get a reliable ROI read."
        )

    top_genre = money_df.groupby("primary_genre")["revenue"].sum().idxmax()

    recommendations.append(
        f"Prioritize {top_genre} -- it's the top-grossing genre in this slice."
    )

    top_language = profit_by_language.idxmax()

    recommendations.append(
        f"Expand distribution in {top_language}-language markets -- currently the most profitable."
    )

    for rec in recommendations:
        st.success(rec)

    # ==========================================
    # Automated Insights
    # ==========================================

    st.header("📊 Automated Insights")

    # Top Grossing Genre
    top_genre_revenue = money_df.groupby("primary_genre")["revenue"].sum().max()

    st.write(f"🏆 **Top Grossing Genre:** {top_genre}")
    st.write(f"💰 **Genre Revenue:** ${top_genre_revenue:,.2f}")

    # Top Language
    top_language_profit = profit_by_language.max()

    st.write(f"🌍 **Most Profitable Language:** {top_language}")
    st.write(f"💵 **Language Profit:** ${top_language_profit:,.2f}")

    # Top Movie
    top_movie = top_movies.index[0]
    top_movie_revenue = top_movies.iloc[0]

    st.write(f"🎬 **Top Grossing Movie:** {top_movie}")
    st.write(f"💲 **Movie Revenue:** ${top_movie_revenue:,.2f}")

    # Most Profitable Genre
    profit_by_genre = (
        money_df.assign(profit=lambda d: d["revenue"] - d["budget"])
        .groupby("primary_genre")["profit"]
        .sum()
    )
    profit_genre = profit_by_genre.idxmax()
    profit_value = profit_by_genre.max()

    st.write(f"📈 **Most Profitable Genre:** {profit_genre}")
    st.write(f"💸 **Profit:** ${profit_value:,.2f}")

    # ==========================================
    # Business Status
    # ==========================================

    st.header("📌 Business Status")

    if total_profit > 0:
        st.success("Portfolio is profitable overall.")
    else:
        st.error("Portfolio is operating at a loss overall.")

    if average_rating < 6:
        st.warning("Average audience rating is below a strong threshold (6/10).")

    if average_runtime > 150:
        st.warning("Average runtime is high -- may affect audience engagement.")

    if pct_budget_known < 30:
        st.warning("Budget/revenue data is unknown for most titles in this slice.")


    with st.expander("📄 View Dataset"):
        st.dataframe(filtered_df)


elif page == "ML Prediction":

    st.header("🤖 ML Prediction")

    budget_input = st.number_input("Budget ($)", min_value=0, value=20000000, step=1000000)

    runtime_input = st.number_input("Runtime (minutes)", min_value=0, value=100)

    release_year_input = st.number_input("Release Year", min_value=1900, max_value=2030, value=2020)

    franchise_input = st.checkbox("Part of a franchise / collection (e.g. a sequel)?")

    genre_input = st.selectbox(
        "Primary Genre",
        sorted(df["primary_genre"].unique().tolist())
    )

    language_input = st.selectbox(
        "Original Language",
        sorted(df["original_language"].dropna().unique().tolist())
    )

    if st.button("Predict"):
        input_data = pd.DataFrame({
            "budget": [budget_input],
            "runtime": [runtime_input],
            "release_year": [release_year_input],
            "is_franchise": [int(franchise_input)],
            "primary_genre": [genre_input],
            "original_language": [language_input]
        })

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        if prediction[0] == 1:
            st.success(
                f"Profitable -- Probability: {probability[0][1]:.2%}"
            )
        else:
            st.error(
                f"Not Profitable -- Probability: {probability[0][0]:.2%}"
            )
