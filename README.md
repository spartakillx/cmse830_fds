# 🏀 NBA Analytics & Hall of Fame Index Dashboard (Final Project)

## Overview
This dashboard unifies **three NBA datasets** to analyze player production, team performance, and accolades, and to derive a **Hall-of-Fame Index** (0–100) that blends longevity, box-score totals, team success, and awards. On top of interactive EDA and player tools, the app includes a **modeling suite** with:
- **Regression** to predict **team win percentage** from team-season features, and  
- **Classification** to predict whether a player is **“Elite”** (HoF Index ≥ configurable threshold).

Built with **Python**, **Streamlit**, **Pandas**, **NumPy**, **Matplotlib/Seaborn**, and **scikit-learn** with caching and session state.

---

## What’s New vs. Midterm
- ✅ **Multiple datasets integrated** (box scores, season records, accolades)  
- ✅ **Robust cleaning & joins** (name normalization, canonical name keys, team abbreviation mapping)  
- ✅ **Accolade-driven Hall-of-Fame Index** with transparent feature contributions  
- ✅ **Two ML tasks** with train/test split, cross-validation, metrics, and exportable predictions  
- ✅ **Richer EDA and player tools** (explorer, comparison, per-season trends)

---

## Data Sources
1. **Player box scores:** `szymonjwiak/nba-traditional`  
2. **Team season records:** `boonpalipatana/nba-season-records-from-every-year`  
3. **Accolades & honors:** `ryanschubertds/all-nba-aba-players-bio-stats-accolades`  

> The app auto-downloads via KaggleHub when possible; it can also read local CSVs from a provided zip.

---

## Key Features
### 1) Data Integration & Cleaning
- Season parsing (e.g., “2016-17” → `2016`)
- **Team name normalization** → abbreviations (e.g., “Golden State Warriors” → `GSW`)
- **Player name normalization** and canonical **name_key** to resolve near-matches (fallback join for accolades)
- Season-level & career-level tables with totals and average team win%

### 2) Exploratory Data Analysis (EDA)
- Dataset overview & coverage diagnostics (win% join rate)  
- Correlation heatmaps, distributions, and league trend lines  
- Scatterplots: player stats vs **team win%**  

### 3) Player Tools
- **Player Explorer:** per-season stats, trends, and rolling windows  
- **Player Comparison:** side-by-side career totals, win%, accolades, HoF Index, and **Top HoF contributors**

### 4) Hall-of-Fame Index (0–100)
- Z-score standardization across production & accolade features  
- Transparent weights (e.g., **MVP=15**, **Finals MVP=12**, **Championships=8**, **All-NBA 1st=4**, etc.)  
- Interactive filters (min seasons/games, top-N), contribution plots, CSV export

### 5) Modeling Suite
**A) Team Win% Regression**  
- Models: Linear Regression (with scaling) & Random Forest  
- Metrics: **RMSE / MAE / R²**, residual plots, permutation importances, **K-Fold CV**

**B) Elite HoF Classification**  
- Label: Elite if **HoF Index ≥ threshold** (user-controlled)  
- Models: Logistic Regression (with scaling) & Random Forest  
- Metrics: **Accuracy / Precision / Recall / F1 / ROC-AUC**, ROC curves, confusion matrix, **Stratified K-Fold CV**

---

## Project Structure
```
nba-analytics-hof/
├─ app.py                        # Streamlit application
├─ README.md                     # This document
├─ requirements.txt              # Dependencies
├─ data/                         # (optional) local csv/zip caches
└─ images/                       # (optional) screenshots/plots
```
---

## How to Run
1) **Install dependencies**
```bash
pip install -r requirements.txt
# or
pip install streamlit pandas numpy matplotlib seaborn scikit-learn kagglehub
```

2) **Run the app**
```bash
streamlit run app.py
```

> If you can’t access Kaggle from your environment, place the three datasets’ CSVs inside a zip under `data/` and the app will attempt local loading automatically.

---

## Reproducibility Notes
- Deterministic seeds for model splits and Random Forests  
- `@st.cache_data` used to cache data loads & feature tables  
- Download buttons export model predictions and curated career tables (with HoF Index)

---

## Rubric Alignment (at a glance)
- **Data**: 3 sources; advanced cleaning; complex joins (name_key fallback + abbr mapping)  
- **EDA**: ≥5 visualization types; statistical summaries; trend lines  
- **Feature Engineering**: per-game rates, aggregates, name keys, team mapping, HoF contributions  
- **Modeling**: 2 tasks × 2 models; metrics, CV, comparisons, feature importances  
- **Streamlit**: ≥5 interactive controls; caching; session state; CSV exports; robust UX  
- **Documentation**: This README + in-app labels and diagnostics

---

## Author
**Aditya Sudarsan Anand**  
CMSE 830 — Final Project

---

## Future Improvements
- Add **hyperparameter tuning** (Optuna / GridSearchCV)  
- Enrich accolades with **All-Star game MVP**, **MIP**, **6MOY** for broader index signals  
- Expand to **play-by-play / RAPM**-style features and **time-series** team modeling  
- Add **Plotly** interactivity and downloadable player reports  
- Containerize with **Docker** and provide CI for reproducible deployments

---

## Notes
- Accolade weights are documented inside the app and can be adjusted in code.  
- All numeric and non-numeric columns used in joins are normalized and validated; join coverage is reported in the Overview.  
- The modeling suite avoids leakage by training on engineered features derived from season/career aggregates.
