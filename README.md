# Movies Dataset KPI & Profitability Analysis

Analysis, dashboard, and a machine learning model built on [The Movies
Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
(TMDB / Kaggle, 45,463 movies, CC0).

## What's in here

- **KPI analysis** — revenue, profit, ratings, runtime, top genres,
  release trends, budget-vs-revenue relationships
- **Interactive dashboard** — filterable KPIs and charts, built with Streamlit
- **ML prediction** — a Random Forest model that predicts whether a movie
  will be profitable, with a live prediction page in the dashboard

## Project structure

```
├── main.py                    # generates static charts + KPI card image
├── src/
│   ├── load_data.py             # loads and cleans the raw CSV
│   ├── kpis.py                  # computes the headline KPI numbers
│   ├── visualize.py              # the 5 analysis charts
│   └── cards.py                  # renders the KPI cards dashboard image
├── movies_kpi_analysis.ipynb   # notebook version of the same analysis
├── kpi_dashboard.py            # Streamlit dashboard + ML Prediction page
├── train_model_project.py     # trains and saves the Random Forest model
├── PROJECT_STACK_SIMPLE.txt   # plain-English list of tools/algorithms used
└── .gitignore
```

`data/` (the CSV) and `model.pkl` (the trained model) aren't tracked in
this repo -- see Setup below to regenerate them.

## Setup

1. Clone the repo
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn streamlit scikit-learn joblib
   ```
3. Download `movies_metadata.csv` from
   [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
   and place it in a `data/` folder in the repo root:
   ```
   data/movies_metadata.csv
   ```
4. Train the model (creates `model.pkl`):
   ```bash
   python train_model_project.py
   ```
5. Run the dashboard:
   ```bash
   streamlit run kpi_dashboard.py
   ```

## Key numbers

- **45,463 movies**, 1930-2020s coverage
- Avg rating: **5.62 / 10** · Avg runtime: **94 min**
- Of movies with known budget & revenue: median ROI **2.06x**,
  **~70% profitable**
- Top genre by movie count and revenue: **Drama** / **Action**
- Prediction model: **~70% accuracy**, ROC AUC **0.70**, predicting
  profitability from budget, runtime, release year, franchise status,
  genre, and language

## A data quirk worth knowing

`budget` and `revenue` are `0` for roughly 80% of rows -- in this dataset
that means "not recorded," not a literal $0 movie. Every KPI and the
model training step filter these out rather than treating them as real
financial data.

## License

Dataset is CC0 (public domain), courtesy of Rounak Banik / TMDB via Kaggle.
