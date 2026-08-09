# -*- coding: utf-8 -*-
"""train_model.py

Movies Dataset -- Classification Model
Predict Profitable Movie
"""

# ==========================================
# Movies Analytics
# Classification Model
# Predict Profitable Movie
# ==========================================

# ==========================================
# Import Libraries
# ==========================================

import ast

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from joblib import dump


# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("data/movies_metadata.csv", low_memory=False)

print("Dataset Loaded Successfully!")


# ==========================================
# Data Cleaning
# ==========================================

df.drop_duplicates(inplace=True)

for col in ["budget", "popularity", "id"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["id"])

df["release_date"] = pd.to_datetime(
    df["release_date"], format="mixed", errors="coerce"
)

df["release_year"] = df["release_date"].dt.year


def parse_genres(raw):
    try:
        return [g["name"] for g in ast.literal_eval(raw)]
    except (ValueError, SyntaxError, TypeError):
        return []


df["genre_list"] = df["genres"].apply(parse_genres)

df["primary_genre"] = df["genre_list"].apply(
    lambda genres: genres[0] if len(genres) > 0 else "Unknown"
)

# belongs_to_collection is only populated for franchise/sequel titles --
# franchises are a known strong signal for reliable box office return
df["is_franchise"] = df["belongs_to_collection"].notna().astype(int)

# 0 in budget/revenue means "not recorded" in this dataset, not a real $0
# movie -- training on those rows would teach the model that almost every
# movie loses money, so only rows with real financial data are kept
df = df[(df["budget"] > 0) & (df["revenue"] > 0)]

df = df.dropna(subset=["runtime", "primary_genre", "original_language", "release_year"])

print("Cleaning Completed Successfully!")


# ==========================================
# Create Target Column
# ==========================================

df["Profitable"] = (df["revenue"] > df["budget"]).astype(int)

print("\nTarget Distribution\n")

print(df["Profitable"].value_counts())


# ==========================================
# Feature Selection
# ==========================================

X = df[
    [
        "budget",
        "runtime",
        "release_year",
        "is_franchise",
        "primary_genre",
        "original_language"
    ]
]

y = df["Profitable"]


# ==========================================
# Categorical Features
# ==========================================

categorical_features = [

    "primary_genre",

    "original_language"

]

preprocessor = ColumnTransformer(

    transformers=[

        (

            "cat",

            OneHotEncoder(handle_unknown="ignore"),

            categorical_features

        )

    ],

    remainder="passthrough"

)


# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


# ==========================================
# Build Pipeline
# ==========================================

model = Pipeline(

    steps=[

        ("preprocessor", preprocessor),

        (

            "classifier",

            RandomForestClassifier(

                n_estimators=200,

                random_state=42

            )

        )

    ]

)


# ==========================================
# Train Model
# ==========================================

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")


# ==========================================
# Prediction
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# Evaluation
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========\n")

print(f"Accuracy  : {accuracy:.4f}")

print(f"Precision : {precision:.4f}")

print(f"Recall    : {recall:.4f}")

print(f"F1 Score  : {f1:.4f}")

print("\nConfusion Matrix\n")

print(cm)

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))


# ==========================================
# Save Model
# ==========================================

dump(model, "model.pkl")

print("\nModel Saved Successfully!")

print("File Name : model.pkl")

from sklearn.metrics import roc_auc_score

y_prob = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)

print(f"\nROC AUC Score : {auc:.4f}")
