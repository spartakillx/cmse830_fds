# app.py
# ==================================================
# CMSE 830 FINAL PROJECT
# NBA MULTI-DATASET ANALYTICS + HOF INDEX + MODELING
# Datasets used ONLY:
#   1) szymonjwiak/nba-traditional (boxscores)
#   2) boonpalipatana/nba-season-records-from-every-year (team win%)
#   3) ryanschubertds/all-nba-aba-players-bio-stats-accolades (awards)
# ==================================================

from __future__ import annotations

import os
import zipfile
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

# ML
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix
)
from sklearn.inspection import permutation_importance

# ------------------------- PAGE -------------------------
st.set_page_config(page_title="NBA Analytics + HoF Index + Modeling", page_icon="🏀", layout="wide")
st.title("NBA Analytics & Hall of Fame Index Dashboard (Accolade-Driven + Modeling)")
st.caption("Sources: szymonjwiak/nba-traditional • boonpalipatana/nba-season-records-from-every-year • ryanschubertds/all-nba-aba-players-bio-stats-accolades")

# ------------------------- SESSION -------------------------
if "last_models" not in st.session_state:
    st.session_state.last_models = {}

# ------------------------- ABOUT CONTENT (single source of truth) -------------------------
ABOUT_MD = """
# 🏀 NBA Analytics & Hall of Fame Index Dashboard (Final Project)

## Overview
This dashboard unifies **three NBA datasets** to analyze player production, team performance, and accolades, and to derive a **Hall-of-Fame Index** (0–100) that blends longevity, box-score totals, team success, and awards. On top of interactive EDA and player tools, the app includes a **modeling suite** with:
- **Regression** to predict **team win percentage** from team-season features, and  
- **Classification** to predict whether a player is **“Elite”** (HoF Index ≥ configurable threshold).

Built with **Python**, **Streamlit**, **Pandas**, **NumPy**, **Matplotlib/Seaborn**, and **scikit-learn** with caching and session state.

## What’s New vs. Midterm
- Multiple datasets integrated (box scores, season records, accolades)
- Robust cleaning & joins (name normalization, canonical name keys, team abbreviation mapping)
- Accolade-driven Hall-of-Fame Index with transparent feature contributions
- Two ML tasks with train/test split, cross-validation, metrics, and exportable predictions
- Richer EDA and player tools (explorer, comparison, per-season trends)

## Data Sources
1. Player box scores: szymonjwiak/nba-traditional
2. Team season records: boonpalipatana/nba-season-records-from-every-year
3. Accolades & honors: ryanschubertds/all-nba-aba-players-bio-stats-accolades

The app auto-downloads via KaggleHub when possible; it can also read local CSVs from a provided zip.

## Key Features
- Data Integration & Cleaning: season parsing, team name normalization to abbreviations, player name normalization + canonical name_key, season/career tables.
- EDA: overview & coverage, heatmaps, distributions, trend lines, scatter vs win%.
- Player Tools: season explorer, comparison with HoF contributions.
- HoF Index (0–100): z-scores across production & accolades with documented weights; percentile scaling; contribution plots; CSV export.
- Modeling Suite: team Win% regression (LR, RF) + Elite HoF classification (LogReg, RF) with metrics, CV, ROC, confusion matrix, importances.

## How to Run
pip install -r requirements.txt
streamlit run app.py

If Kaggle is blocked, place the three datasets’ CSVs inside a zip under data/ and the app will try local loading.

## Reproducibility
- Deterministic seeds
- @st.cache_data for data/features
- Download buttons for predictions and curated career tables (with HoF Index)

## Rubric Alignment
- Data: 3 sources; advanced cleaning; complex joins (name_key fallback + abbr mapping)
- EDA: multiple visualization types; statistical summaries; trend lines
- Feature Engineering: per-game rates, aggregates, name keys, team mapping, HoF contributions
- Modeling: 2 tasks × 2 models; metrics, CV, comparisons, importances
- Streamlit: multiple interactives; caching; session state; CSV exports; robust UX
- Documentation: README + in-app guidance

## Author
Aditya Sudarsan Anand — CMSE 830 Final Project
"""

def render_about_tab() -> None:
    st.markdown(ABOUT_MD)
    # Prefer a cached README if present; otherwise use ABOUT_MD
    candidates = [
        "/mnt/data/FINAL_README.md",
        "README.md",
    ]
    data_bytes = None
    file_name = "README.md"
    for p in candidates:
        if os.path.exists(p):
            with open(p, "rb") as fh:
                data_bytes = fh.read()
                file_name = os.path.basename(p)
                break
    if data_bytes is None:
        data_bytes = ABOUT_MD.encode("utf-8")
        file_name = "README.md"
    st.download_button(
        "Download README",
        data=data_bytes,
        file_name=file_name,
        mime="text/markdown",
        use_container_width=True,
    )

# Persist a copy for download
try:
    os.makedirs("/mnt/data", exist_ok=True)
    with open("/mnt/data/FINAL_README.md", "w", encoding="utf-8") as _f:
        _f.write(ABOUT_MD)
except Exception:
    pass

# ------------------------- CONSTANTS -------------------------
MIN_YEAR = 2005
LOCAL_ZIP_GLOBS = ["/mnt/data/archive.zip", "/mnt/data/archive (1).zip", "/mnt/data/archive (2).zip"]

ACC_MAPPING: Dict[str, List[str]] = {
    "championships": ["championships", "rings", "titles", "champion"],
    "mvp": ["mvp", "mvps", "league_mvp", "season_mvp", "nba mvp"],
    "finals_mvp": ["finals_mvp", "fmvp", "finals_mvps", "finals mvp"],
    "all_nba_first": ["all_nba_first", "all-nba_first", "all_nba_1st", "all-nba 1st team"],
    "all_nba_second": ["all_nba_second", "all-nba_second", "all_nba_2nd", "all-nba 2nd team"],
    "all_nba_third": ["all_nba_third", "all-nba_third", "all_nba_3rd", "all-nba 3rd team"],
    "all_star": ["all_star_count", "all-star count", "all_star", "allstar", "all_stars", "all-star"],
    "dpoy": ["dpoy", "defensive_player_of_the_year", "dpoy_awards", "defensive player of the year"],
    "all_defensive_first": ["all_defensive_first", "all-defensive-first", "all_def_1st", "all-defense 1st team"],
    "all_defensive_second": ["all_defensive_second", "all-defensive-second", "all_def_2nd", "all-defense 2nd team"],
    "roy": ["roy", "rookie_of_the_year", "rookie of the year"],
    "scoring_titles": ["scoring_champion", "scoring_titles", "scoring_leader", "scoring champion"],
}

HOF_WEIGHTS: Dict[str, float] = {
    "seasons": 1.0, "games": 0.6, "tot_pts": 0.7, "tot_reb": 0.35, "tot_ast": 0.45, "avg_team_win_pct": 1.2,
    "mvp": 15.0, "finals_mvp": 12.0, "championships": 8.0, "dpoy": 5.0,
    "all_nba_first": 4.0, "all_nba_total": 2.5, "all_star": 2.0,
    "all_defensive_first": 2.0, "all_defensive_total": 1.0,
    "roy": 1.5, "scoring_titles": 1.5,
}

ALL_POSSIBLE_ACCS = [
    "mvp","finals_mvp","championships","dpoy","all_nba_first","all_nba_second","all_nba_third","all_nba_total",
    "all_star","all_defensive_first","all_defensive_second","all_defensive_total","roy","scoring_titles"
]

# ------------------------- TEAM ABBR MAP -------------------------
TEAM_NAME_TO_ABBR: Dict[str, str] = {
    "ATLANTA HAWKS": "ATL", "BOSTON CELTICS": "BOS", "BROOKLYN NETS": "BKN", "NEW JERSEY NETS": "NJN",
    "CHARLOTTE BOBCATS": "CHA", "CHARLOTTE HORNETS": "CHA", "CHICAGO BULLS": "CHI", "CLEVELAND CAVALIERS": "CLE",
    "DETROIT PISTONS": "DET", "INDIANA PACERS": "IND", "MIAMI HEAT": "MIA", "MILWAUKEE BUCKS": "MIL",
    "NEW YORK KNICKS": "NYK", "ORLANDO MAGIC": "ORL", "PHILADELPHIA 76ERS": "PHI", "TORONTO RAPTORS": "TOR",
    "WASHINGTON WIZARDS": "WAS",
    "DALLAS MAVERICKS": "DAL", "DENVER NUGGETS": "DEN", "GOLDEN STATE WARRIORS": "GSW", "HOUSTON ROCKETS": "HOU",
    "LA CLIPPERS": "LAC", "LOS ANGELES CLIPPERS": "LAC", "LOS ANGELES LAKERS": "LAL", "MEMPHIS GRIZZLIES": "MEM",
    "MINNESOTA TIMBERWOLVES": "MIN", "NEW ORLEANS HORNETS": "NOH", "NEW ORLEANS/OKLAHOMA CITY HORNETS": "NOK",
    "NEW ORLEANS PELICANS": "NOP", "OKLAHOMA CITY THUNDER": "OKC", "SEATTLE SUPERSONICS": "SEA",
    "PHOENIX SUNS": "PHX", "PORTLAND TRAIL BLAZERS": "POR", "SACRAMENTO KINGS": "SAC", "SAN ANTONIO SPURS": "SAS",
    "UTAH JAZZ": "UTA",
    "GS WARRIORS": "GSW", "SA SPURS": "SAS", "NO PELICANS": "NOP", "NO HORNETS": "NOH"
}

def to_abbr(team_val: str) -> str:
    if not isinstance(team_val, str):
        return ""
    s = team_val.strip().upper()
    if 2 <= len(s) <= 4 and s.isalpha():
        return s
    s = s.replace(".", " ").replace("-", " ").replace("’", "'").replace("'", "")
    s = " ".join(s.split())
    return TEAM_NAME_TO_ABBR.get(s, s)

# ------------------------- HELPERS -------------------------
def _safe_to_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None

def _season_to_start_year(x) -> Optional[int]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if "-" in s:
        s = s.split("-")[0]
    return _safe_to_int(s)

def _normalize_names(series: pd.Series) -> pd.Series:
    return (series.fillna("").astype(str).str.strip()
            .str.replace(r"\.", "", regex=True).str.replace(r"\s+", " ", regex=True).str.title())

def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def safe_int_slider(label: str, min_value: int, max_value: int, value: int, step: int = 1, key: Optional[str] = None) -> int:
    if max_value is None or min_value is None:
        return value
    max_value = int(max_value)
    min_value = int(min_value)
    value = int(value)
    if max_value < min_value:
        max_value = min_value
    if max_value == min_value:
        st.number_input(label, min_value=min_value, max_value=max_value, value=value, step=1, key=key, disabled=True)
        return value
    step = max(1, int(step))
    value = min(max(value, min_value), max_value)
    return st.slider(label, min_value=min_value, max_value=max_value, value=value, step=step, key=key)

def canonical_name_key(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.strip().lower()
    s = "".join(ch for ch in s if ch.isalpha() or ch.isspace())
    tokens = [t for t in s.split() if t]
    suffixes = {"jr", "sr", "ii", "iii", "iv", "v"}
    tokens = [t for t in tokens if t not in suffixes]
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]
    return f"{tokens[0]} {tokens[-1]}"

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# ------------------------- LOADING -------------------------
def _list_csvs_in_zip(zip_path: str) -> List[str]:
    if not os.path.exists(zip_path):
        return []
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            return [zi.filename for zi in z.infolist() if zi.filename.lower().endswith(".csv")]
    except Exception:
        return []

def _read_csv_from_zip(zip_path: str, name_hint: Optional[str] = None) -> Optional[pd.DataFrame]:
    if not os.path.exists(zip_path):
        return None
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            names = _list_csvs_in_zip(zip_path)
            target = None
            if name_hint:
                for n in names:
                    if name_hint.lower() in n.lower():
                        target = n
                        break
            if target is None and names:
                target = names[0]
            if target:
                with z.open(target) as f:
                    return pd.read_csv(f)
    except Exception:
        return None
    return None

def _load_kaggle_csv_any(dataset_id: str) -> pd.DataFrame:
    path = kagglehub.dataset_download(dataset_id)
    files = [f for f in os.listdir(path) if f.lower().endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"No CSV in {dataset_id}")
    files = sorted(files, key=lambda f: os.path.getsize(os.path.join(path, f)), reverse=True)
    return pd.read_csv(os.path.join(path, files[0]))

@st.cache_data(show_spinner=True)
def load_boxscores() -> pd.DataFrame:
    try:
        df = _load_kaggle_csv_any("szymonjwiak/nba-traditional")
    except Exception as e:
        st.warning(f"Kaggle boxscores unavailable ({e}); trying local archive...")
        df = None
        for zp in LOCAL_ZIP_GLOBS:
            df = _read_csv_from_zip(zp, "traditional") or _read_csv_from_zip(zp, "game")
            if df is not None:
                break
        if df is None:
            raise
    df.columns = df.columns.str.strip().str.lower()
    rename = {}
    season_col = _first_existing_column(df, ["season", "season_id"])
    player_col = _first_existing_column(df, ["player_name", "player"])
    team_col = _first_existing_column(df, ["team_abbreviation", "team"])
    if season_col:
        rename[season_col] = "season"
    if player_col:
        rename[player_col] = "player_name"
    if team_col:
        rename[team_col] = "team"
    if "game_id" in df.columns:
        rename["game_id"] = "game_id"
    for stat in ["pts", "reb", "ast", "stl", "blk", "tov", "min"]:
        if stat in df.columns:
            rename[stat] = stat
    df = df.rename(columns=rename)
    keep = [c for c in ["season", "player_name", "team", "game_id", "pts", "reb", "ast", "stl", "blk"] if c in df.columns]
    df = df[keep].copy()
    if "season" in df.columns:
        df["year"] = df["season"].apply(_season_to_start_year)
        df = df.dropna(subset=["year"])
        df["year"] = df["year"].astype(int)
        df = df[df["year"] >= MIN_YEAR]
    if "player_name" in df.columns:
        df["player_name"] = _normalize_names(df["player_name"])
    if "team" in df.columns:
        df["team"] = df["team"].str.upper()
    return df

@st.cache_data(show_spinner=True)
def load_team_records() -> pd.DataFrame:
    def _season_to_year_any(s) -> Optional[int]:
        if pd.isna(s):
            return None
        s = str(s).strip()
        if "-" in s:
            return _safe_to_int(s.split("-")[0])
        return _safe_to_int(s)

    try:
        df = _load_kaggle_csv_any("boonpalipatana/nba-season-records-from-every-year")
    except Exception as e:
        st.warning(f"Kaggle team records unavailable ({e}); trying local archive...")
        df = None
        for zp in LOCAL_ZIP_GLOBS:
            df = _read_csv_from_zip(zp, "season") or _read_csv_from_zip(zp, "records")
            if df is not None:
                break
        if df is None:
            return pd.DataFrame(columns=["year", "team_abbr", "win_pct"])

    df.columns = df.columns.str.strip().str.lower()
    year_col = _first_existing_column(df, ["year", "season", "season_end", "season_start"])
    team_col = _first_existing_column(df, ["team", "team_name", "franchise", "club", "name"])
    wins_col = _first_existing_column(df, ["wins", "w"])
    losses_col = _first_existing_column(df, ["losses", "l"])
    winpct_col = _first_existing_column(df, ["win_pct", "win%", "w/l%", "win percent", "win percentage"])

    work = pd.DataFrame()
    if year_col is not None:
        work["year"] = df[year_col].apply(_season_to_year_any)
    if team_col is not None:
        work["team"] = df[team_col]
    if wins_col is not None:
        work["wins"] = pd.to_numeric(df[wins_col], errors="coerce")
    if losses_col is not None:
        work["losses"] = pd.to_numeric(df[losses_col], errors="coerce")
    if winpct_col is not None:
        work["win_pct"] = pd.to_numeric(df[winpct_col], errors="coerce")
    if "win_pct" not in work.columns and {"wins", "losses"}.issubset(work.columns):
        denom = (work["wins"] + work["losses"]).replace(0, np.nan)
        work["win_pct"] = (work["wins"] / denom).astype(float)

    if "team" in work.columns:
        work["team"] = work["team"].astype(str).str.strip()
        work["team_abbr"] = work["team"].apply(to_abbr).str.upper()

    if "year" in work.columns:
        work = work.dropna(subset=["year"])
        work["year"] = work["year"].astype(int)
        work = work[work["year"] >= MIN_YEAR]

    keep = [c for c in ["year", "team", "team_abbr", "win_pct"] if c in work.columns]
    if not keep:
        return pd.DataFrame(columns=["year", "team_abbr", "win_pct"])
    return work[keep].dropna(subset=["team_abbr"]).drop_duplicates()

@st.cache_data(show_spinner=True)
def load_accolades() -> pd.DataFrame:
    try:
        df = _load_kaggle_csv_any("ryanschubertds/all-nba-aba-players-bio-stats-accolades")
    except Exception as e:
        st.warning(f"Kaggle accolades unavailable ({e}); trying local archive...")
        df = None
        for zp in LOCAL_ZIP_GLOBS:
            df = _read_csv_from_zip(zp, "accolade") or _read_csv_from_zip(zp, "all-nba")
            if df is not None:
                break
        if df is None:
            return pd.DataFrame(columns=["player_name", "name_key"])

    df.columns = df.columns.str.strip().str.lower()
    name_col = _first_existing_column(df, ["player", "player_name", "name"])
    if name_col:
        df = df.rename(columns={name_col: "player_name"})
    else:
        return pd.DataFrame(columns=["player_name", "name_key"])

    df["player_name"] = _normalize_names(df["player_name"])

    standard_cols = {}
    for canonical, candidates in ACC_MAPPING.items():
        for c in candidates:
            if c in df.columns:
                standard_cols[c] = canonical
                break
    df = df.rename(columns=standard_cols)

    acc_cols = [c for c in ACC_MAPPING.keys() if c in df.columns]
    if not acc_cols:
        agg = df[["player_name"]].drop_duplicates()
        agg["name_key"] = agg["player_name"].apply(canonical_name_key)
        return agg

    for c in acc_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    agg = df.groupby("player_name", as_index=False)[acc_cols].sum()

    if set(["all_nba_first", "all_nba_second", "all_nba_third"]).intersection(agg.columns):
        cols = [c for c in ["all_nba_first", "all_nba_second", "all_nba_third"] if c in agg.columns]
        agg["all_nba_total"] = agg[cols].sum(axis=1)
    if set(["all_defensive_first", "all_defensive_second"]).intersection(agg.columns):
        cols = [c for c in ["all_defensive_first", "all_defensive_second"] if c in agg.columns]
        agg["all_defensive_total"] = agg[cols].sum(axis=1)

    agg["name_key"] = agg["player_name"].apply(canonical_name_key)
    return agg

# ------------------------- BUILD TABLES -------------------------
@st.cache_data(show_spinner=True)
def build_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    box = load_boxscores()
    teams = load_team_records()
    acc = load_accolades()

    # Player-season aggregates
    if box.empty or "year" not in box.columns or "player_name" not in box.columns:
        season_df = pd.DataFrame()
    else:
        agg_dict = {}
        for c in ["pts", "reb", "ast", "stl", "blk"]:
            if c in box.columns:
                agg_dict[c] = "sum"
        if "game_id" in box.columns:
            agg_dict["games"] = pd.Series.nunique
        gb_cols = [c for c in ["year", "player_name", "team"] if c in box.columns]
        season_df = box.groupby(gb_cols, as_index=False).agg(agg_dict)

    # Merge team win% using abbreviations
    if not season_df.empty and not teams.empty and {"year", "team_abbr"}.issubset(teams.columns):
        join_df = teams[["year", "team_abbr", "win_pct"]].rename(columns={"team_abbr": "team"})
        season_df = season_df.merge(join_df, on=["year", "team"], how="left")

    # Career aggregate
    if season_df.empty:
        career_df = pd.DataFrame(columns=["player_name"])
    else:
        career_agg: Dict[str, Tuple[str, str]] = {
            "from_year": ("year", "min"),
            "to_year": ("year", "max"),
            "seasons": ("year", "nunique"),
        }
        if "games" in season_df.columns:
            career_agg["games"] = ("games", "sum")
        for c in ["pts", "reb", "ast", "stl", "blk"]:
            if c in season_df.columns:
                career_agg[f"tot_{c}"] = (c, "sum")
        if "win_pct" in season_df.columns:
            career_agg["avg_team_win_pct"] = ("win_pct", "mean")
        career_df = season_df.groupby("player_name").agg(**career_agg).reset_index()

    # Name key on career
    if not career_df.empty and "player_name" in career_df.columns:
        career_df["name_key"] = career_df["player_name"].apply(canonical_name_key)

    # Accolades merge (exact, then fallback by name_key via map)
    award_cols_set = set(ACC_MAPPING.keys()) | {"all_nba_total", "all_defensive_total"}
    award_cols = [c for c in acc.columns if c in award_cols_set]
    if not acc.empty and not career_df.empty:
        cols_for_exact = ["player_name"] + (["name_key"] if "name_key" in acc.columns else []) + award_cols
        merged = career_df.merge(acc[cols_for_exact], on="player_name", how="left", suffixes=("", "_acc"))
        if award_cols:
            need_fill = merged[award_cols].isna().all(axis=1)
            if need_fill.any() and "name_key" in merged.columns and "name_key" in acc.columns:
                acc_key = acc.groupby("name_key")[award_cols].max()
                key_series = merged.loc[need_fill, "name_key"]
                for col in award_cols:
                    merged.loc[need_fill, col] = key_series.map(acc_key[col]).values
        career_df = merged

    # Fill awards to zero
    for c in award_cols:
        career_df[c] = pd.to_numeric(career_df[c], errors="coerce").fillna(0)

    # Coerce numerics
    for c in ["from_year", "to_year", "seasons", "games", "tot_pts", "tot_reb", "tot_ast", "tot_stl", "tot_blk"]:
        if c in career_df.columns:
            career_df[c] = pd.to_numeric(career_df[c], errors="coerce")
    if "avg_team_win_pct" in career_df.columns:
        career_df["avg_team_win_pct"] = pd.to_numeric(career_df["avg_team_win_pct"], errors="coerce")

    return season_df, teams, career_df

season_df, team_df, career_df = build_tables()

# ------------------------- HOF INDEX -------------------------
def compute_hof_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["player_name", "hof_index"])
    base_cols = ["seasons", "games", "tot_pts", "tot_reb", "tot_ast", "avg_team_win_pct"]
    acc_cols = ["mvp", "finals_mvp", "championships", "dpoy", "all_nba_first", "all_nba_total",
                "all_star", "all_defensive_first", "all_defensive_total", "roy", "scoring_titles"]
    use_cols = [c for c in base_cols if c in df.columns] + [c for c in acc_cols if c in df.columns]
    if not use_cols:
        out = df.copy()
        out["hof_index"] = 0.0
        return out

    X = df[use_cols].fillna(0.0).astype(float)
    weights = pd.Series({c: HOF_WEIGHTS.get(c, 0.0) for c in use_cols})
    mu = X.mean()
    sigma = X.std(ddof=0).replace(0, 1.0)
    Z = (X - mu) / sigma
    contrib = Z.mul(weights, axis=1)
    raw = contrib.sum(axis=1)
    ranks = raw.rank(method="average", pct=True)

    out = df.copy()
    out["hof_index"] = (ranks * 100).round(1)
    for c in contrib.columns:
        out[f"hof_c_{c}"] = contrib[c]
    out["hof_raw"] = raw
    return out

career_df = compute_hof_index(career_df)
accolade_cols_available = [c for c in ALL_POSSIBLE_ACCS if c in career_df.columns]

# ------------------------- FEATURES FOR MODELING -------------------------
@st.cache_data(show_spinner=True)
def make_team_season_regression(season_df: pd.DataFrame) -> pd.DataFrame:
    if season_df.empty or "win_pct" not in season_df.columns:
        return pd.DataFrame(columns=["year", "team", "win_pct"])
    agg = season_df.groupby(["year", "team"], as_index=False).agg({
        **{c: "sum" for c in [x for x in ["pts", "reb", "ast", "stl", "blk", "games"] if x in season_df.columns]},
        "win_pct": "mean",
    })
    if "games" in agg.columns and agg["games"].gt(0).any():
        for c in ["pts", "reb", "ast", "stl", "blk"]:
            if c in agg.columns:
                agg[f"{c}_per_game"] = agg[c] / agg["games"].replace(0, np.nan)
    return agg.dropna(subset=["win_pct"])

@st.cache_data(show_spinner=True)
def make_hof_classification(career_df: pd.DataFrame, elite_threshold: float = 85.0) -> pd.DataFrame:
    if career_df.empty or "hof_index" not in career_df.columns:
        return pd.DataFrame(columns=["player_name", "elite"])
    base = career_df.copy()
    base["elite"] = (base["hof_index"] >= elite_threshold).astype(int)
    feats = [c for c in ["seasons", "games", "tot_pts", "tot_reb", "tot_ast", "avg_team_win_pct"] if c in base.columns]
    cols = ["player_name", "elite"] + feats
    return base[cols].dropna()

team_season_df = make_team_season_regression(season_df)
hof_clf_df = make_hof_classification(career_df)

# ------------------------- UI TABS -------------------------
tabs = st.tabs([
    "Overview",
    "EDA (Season & Team)",
    "Player Explorer",
    "Player Comparison",
    "Modeling",
    "Hall of Fame Explorer",
    "About",
])

# Overview
with tabs[0]:
    st.subheader("Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Players", f"{len(career_df):,}")
    c2.metric("Players (Season Data)", f"{season_df['player_name'].nunique():,}" if "player_name" in season_df.columns else "0")
    c3.metric("Player-Seasons", f"{len(season_df):,}")
    year_range = f"{int(season_df['year'].min())}-{int(season_df['year'].max())}" if "year" in season_df.columns and not season_df.empty else "N/A"
    c4.metric("Year Range", year_range)

    st.markdown("### Accolades Data Status")
    if accolade_cols_available:
        pretty = {
            "mvp":"MVP","finals_mvp":"Finals MVP","championships":"Championships","dpoy":"DPOY",
            "all_nba_first":"All-NBA 1st","all_nba_second":"All-NBA 2nd","all_nba_third":"All-NBA 3rd",
            "all_nba_total":"All-NBA Total","all_star":"All-Star","all_defensive_first":"All-Defense 1st",
            "all_defensive_second":"All-Defense 2nd","all_defensive_total":"All-Defense Total","roy":"ROY","scoring_titles":"Scoring Titles"
        }
        st.success("Included: " + ", ".join(pretty.get(c, c) for c in accolade_cols_available))
    else:
        st.warning("No accolades columns detected.")

    st.markdown("---")
    la, lb = st.columns(2)
    with la:
        st.markdown("Season-level (sample)")
        st.dataframe(season_df.head(20), use_container_width=True)
    with lb:
        show_cols = [c for c in ["player_name","from_year","to_year","seasons","games","tot_pts","tot_reb","tot_ast","avg_team_win_pct","hof_index"] if c in career_df.columns]
        st.markdown("Career-level (sample)")
        st.dataframe(career_df[show_cols].head(20), use_container_width=True)

    st.markdown("---")
    st.markdown("### Data quality: Win% join")
    if not season_df.empty and "win_pct" in season_df.columns:
        joined = season_df["win_pct"].notna().sum()
        total = len(season_df)
        coverage = (joined / total * 100) if total else 0.0
        m1, m2, m3 = st.columns(3)
        m1.metric("Player-Seasons", f"{total:,}")
        m2.metric("With Win%", f"{joined:,}")
        m3.metric("Coverage", f"{coverage:.1f}%")
        missing = season_df[season_df["win_pct"].isna()]
        if not missing.empty:
            st.caption("Sample unmatched (year/team)")
            st.dataframe(missing[["year","team"]].drop_duplicates().head(15), use_container_width=True)
    else:
        st.info("Win% not present; check dataset availability.")

# EDA
with tabs[1]:
    st.subheader("Season & Team EDA")
    if not season_df.empty:
        numeric_cols = season_df.select_dtypes(include=np.number).columns.tolist()
        ca, cb = st.columns(2)
        with ca:
            if len(numeric_cols) >= 2:
                method = st.selectbox("Correlation method", ["pearson", "spearman"], index=0)
                corr = season_df[numeric_cols].corr(method=method)
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
                ax.set_title(f"Correlation ({method})")
                st.pyplot(fig, use_container_width=True)
        with cb:
            if numeric_cols:
                stat = st.selectbox("Distribution stat", numeric_cols, index=0)
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.histplot(season_df[stat].dropna(), kde=True, bins=30, ax=ax)
                ax.set_title(f"Distribution of {stat}")
                st.pyplot(fig, use_container_width=True)

        st.markdown("---")
        if "year" in season_df.columns:
            stat_to_plot = st.selectbox("Line: choose stat vs year", [c for c in ["pts","reb","ast"] if c in season_df.columns])
            tmp = season_df.groupby("year")[stat_to_plot].sum().reset_index()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(tmp["year"], tmp[stat_to_plot], marker="o")
            ax.set_xlabel("Year"); ax.set_ylabel(stat_to_plot.upper())
            ax.set_title(f"League {stat_to_plot.upper()} totals by year")
            st.pyplot(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("Scatter vs Team Win%")
        stat_choice = st.selectbox("Scatter X", [c for c in ["pts","reb","ast","stl","blk","games"] if c in season_df.columns])
        plot_df = season_df[[stat_choice, "win_pct"]].copy() if "win_pct" in season_df.columns else pd.DataFrame()
        if not plot_df.empty:
            plot_df = plot_df.dropna()
            plot_df = plot_df[(plot_df["win_pct"] > 0) & (plot_df["win_pct"] <= 1)]
        if plot_df.empty:
            st.info("No joined Win% available to plot.")
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=plot_df, x=stat_choice, y="win_pct", alpha=0.35, ax=ax)
            ax.set_title(f"{stat_choice} vs Team Win%"); ax.set_ylim(0, 1)
            st.pyplot(fig, use_container_width=True)

# Player Explorer
with tabs[2]:
    st.subheader("Player explorer (season-level)")
    if "player_name" in season_df.columns and not season_df.empty:
        players = sorted(season_df["player_name"].dropna().unique())
        sel_player = st.selectbox("Select a player", players)
        pdf = season_df[season_df["player_name"] == sel_player].sort_values("year")
        st.dataframe(pdf, use_container_width=True)
        stat_choices = [c for c in ["pts","reb","ast","stl","blk"] if c in pdf.columns]
        if stat_choices:
            stat_to_plot = st.selectbox("Plot stat over time", stat_choices, index=0)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(pdf["year"], pdf[stat_to_plot], marker="o")
            ax.set_xlabel("Year"); ax.set_ylabel(stat_to_plot.upper())
            ax.set_title(f"{sel_player} - {stat_to_plot.upper()} over seasons")
            st.pyplot(fig, use_container_width=True)
        if "pts" in pdf.columns and len(pdf) > 0:
            max_win = int(min(10, len(pdf)))
            default_win = min(3, max_win) if max_win >= 1 else 1
            window = safe_int_slider("Rolling window (seasons)", 1, max_win if max_win >= 1 else 1, default_win, step=1, key="roll_win")
            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(pdf["year"], pdf["pts"], alpha=0.4)
            ax.plot(pdf["year"], pdf["pts"].rolling(window, min_periods=1).mean(), linewidth=2)
            ax.set_title(f"{sel_player} - Points (rolling {window})")
            st.pyplot(fig, use_container_width=True)

# Player Comparison
with tabs[3]:
    st.subheader("Player Comparison")
    if "player_name" in career_df.columns and len(career_df) >= 2:
        plist = sorted(career_df["player_name"].unique())
        c1, c2 = st.columns(2)
        p1 = c1.selectbox("Select Player 1", plist, key="cmp_p1")
        p2 = c2.selectbox("Select Player 2", plist, index=1 if len(plist) > 1 else 0, key="cmp_p2")
        row1 = career_df[career_df["player_name"] == p1].iloc[0]
        row2 = career_df[career_df["player_name"] == p2].iloc[0]
        st.markdown("Career Statistics")
        comp_stats = ["seasons","games","tot_pts","tot_reb","tot_ast","avg_team_win_pct","hof_index"]
        comp_stats += [c for c in accolade_cols_available if c in row1.index and c in row2.index]
        rows = []
        for s in comp_stats:
            v1, v2 = row1.get(s, np.nan), row2.get(s, np.nan)
            if s == "avg_team_win_pct":
                f1 = f"{float(v1):.3f}" if pd.notna(v1) else "NA"
                f2 = f"{float(v2):.3f}" if pd.notna(v2) else "NA"
            elif s == "hof_index":
                f1 = f"{float(v1):.1f}" if pd.notna(v1) else "NA"
                f2 = f"{float(v2):.1f}" if pd.notna(v2) else "NA"
            else:
                f1 = f"{int(v1):,}" if pd.notna(v1) else "NA"
                f2 = f"{int(v2):,}" if pd.notna(v2) else "NA"
            rows.append({"Statistic": s.replace("tot_", "Total ").replace("_", " ").title(), p1: f1, p2: f2})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        contrib_cols = [c for c in career_df.columns if c.startswith("hof_c_")]
        def _plot_contrib(r, title):
            if not contrib_cols:
                return
            contrib = r[contrib_cols].rename(lambda c: c.replace("hof_c_", ""))
            contrib = contrib[contrib.abs().sort_values(ascending=False).index][:10]
            fig, ax = plt.subplots(figsize=(7, 4))
            contrib.plot(kind="barh", ax=ax)
            ax.invert_yaxis()
            ax.set_title(title)
            st.pyplot(fig, use_container_width=True)
        _plot_contrib(row1, f"{p1} - Top HOF contributors")
        _plot_contrib(row2, f"{p2} - Top HOF contributors")

# Modeling
with tabs[4]:
    st.subheader("Modeling Suite: Team Win% Regression and Elite HoF Classification")

    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    cv_k = st.slider("Cross-Validation folds", 3, 10, 5, 1)

    st.markdown("A) Regression: Predict Team Win% from team-season stats")
    if team_season_df.empty:
        st.warning("Team-season dataset unavailable.")
    else:
        reg_feats_all = [c for c in team_season_df.columns if c not in ["year", "team", "win_pct"]]
        default_feats = [c for c in reg_feats_all if c.endswith("_per_game")] or reg_feats_all[:5]
        reg_feats = st.multiselect("Features for regression", reg_feats_all, default=default_feats)
        if reg_feats:
            X = team_season_df[reg_feats].fillna(0.0).values
            y = team_season_df["win_pct"].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

            lr = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
            rfr = RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=-1)

            lr.fit(X_train, y_train)
            rfr.fit(X_train, y_train)

            y_pred_lr = lr.predict(X_test)
            y_pred_rf = rfr.predict(X_test)

            def reg_metrics(y_true, y_pred):
                return {
                    "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                    "MAE": float(mean_absolute_error(y_true, y_pred)),
                    "R2": float(r2_score(y_true, y_pred)),
                }

            m_lr = reg_metrics(y_test, y_pred_lr)
            m_rf = reg_metrics(y_test, y_pred_rf)

            colA, colB = st.columns(2)
            with colA:
                st.markdown("Linear Regression")
                st.json(m_lr)
                kf = KFold(n_splits=cv_k, shuffle=True, random_state=seed)
                cv_r2 = cross_val_score(lr, team_season_df[reg_feats].values, team_season_df["win_pct"].values, cv=kf, scoring="r2")
                st.caption(f"CV R2 mean={cv_r2.mean():.3f} +/- {cv_r2.std():.3f}")
            with colB:
                st.markdown("Random Forest Regressor")
                st.json(m_rf)
                kf = KFold(n_splits=cv_k, shuffle=True, random_state=seed)
                cv_r2 = cross_val_score(rfr, team_season_df[reg_feats].values, team_season_df["win_pct"].values, cv=kf, scoring="r2", n_jobs=-1)
                st.caption(f"CV R2 mean={cv_r2.mean():.3f} +/- {cv_r2.std():.3f}")

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(y_test, y_test - y_pred_lr, alpha=0.4, label="LR")
            ax.scatter(y_test, y_test - y_pred_rf, alpha=0.4, label="RF")
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("True Win%"); ax.set_ylabel("Residual"); ax.set_title("Residuals (test)")
            ax.legend(); st.pyplot(fig, use_container_width=True)

            try:
                perm = permutation_importance(rfr, X_test, y_test, n_repeats=10, random_state=seed, n_jobs=-1)
                imp = pd.Series(perm.importances_mean, index=reg_feats).sort_values(ascending=True).tail(12)
                fig, ax = plt.subplots(figsize=(7, 4))
                imp.plot(kind="barh", ax=ax)
                ax.set_title("Permutation Importances (RF)")
                st.pyplot(fig, use_container_width=True)
            except Exception:
                pass

            st.session_state.last_models["regression"] = {"features": reg_feats, "metrics": {"LR": m_lr, "RF": m_rf}}
            pred_df = pd.DataFrame({"y_true": y_test, "y_pred_lr": y_pred_lr, "y_pred_rf": y_pred_rf})
            st.download_button("Download regression predictions", data=to_csv_bytes(pred_df), file_name="regression_predictions.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("B) Classification: Predict Elite HoF (index >= threshold) from production features")
    if hof_clf_df.empty:
        st.warning("HoF classification dataset unavailable.")
    else:
        thr = st.slider("Elite threshold (HoF Index)", 70.0, 95.0, 85.0, 1.0)
        clf_df = make_hof_classification(career_df, elite_threshold=thr)
        class_feats_all = [c for c in clf_df.columns if c not in ["player_name", "elite"]]
        class_feats = st.multiselect("Features for classification", class_feats_all, default=class_feats_all)
        if class_feats:
            X = clf_df[class_feats].values
            y = clf_df["elite"].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

            logit = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=2000))])
            rfc = RandomForestClassifier(n_estimators=600, random_state=seed, n_jobs=-1, class_weight="balanced")

            logit.fit(X_train, y_train)
            rfc.fit(X_train, y_train)

            proba_lg = logit.predict_proba(X_test)[:, 1]
            proba_rf = rfc.predict_proba(X_test)[:, 1]

            def clf_metrics(y_true, proba):
                pred = (proba >= 0.5).astype(int)
                return {
                    "Accuracy": accuracy_score(y_true, pred),
                    "Precision": precision_score(y_true, pred, zero_division=0),
                    "Recall": recall_score(y_true, pred, zero_division=0),
                    "F1": f1_score(y_true, pred, zero_division=0),
                    "ROC_AUC": roc_auc_score(y_true, proba),
                }

            m_lg = clf_metrics(y_test, proba_lg)
            m_rf = clf_metrics(y_test, proba_rf)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("Logistic Regression")
                st.json({k: round(v, 3) for k, v in m_lg.items()})
                skf = StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=seed)
                cv_auc = cross_val_score(logit, clf_df[class_feats].values, clf_df["elite"].values, cv=skf, scoring="roc_auc")
                st.caption(f"CV ROC AUC mean={cv_auc.mean():.3f} +/- {cv_auc.std():.3f}")
            with col2:
                st.markdown("Random Forest Classifier")
                st.json({k: round(v, 3) for k, v in m_rf.items()})
                skf = StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=seed)
                cv_auc = cross_val_score(rfc, clf_df[class_feats].values, clf_df["elite"].values, cv=skf, scoring="roc_auc", n_jobs=-1)
                st.caption(f"CV ROC AUC mean={cv_auc.mean():.3f} +/- {cv_auc.std():.3f}")

            fpr_lg, tpr_lg, _ = roc_curve(y_test, proba_lg)
            fpr_rf, tpr_rf, _ = roc_curve(y_test, proba_rf)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr_lg, tpr_lg, label=f"LogReg (AUC={m_lg['ROC_AUC']:.3f})")
            ax.plot(fpr_rf, tpr_rf, label=f"RF (AUC={m_rf['ROC_AUC']:.3f})")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve"); ax.legend()
            st.pyplot(fig, use_container_width=True)

            pred_rf = (proba_rf >= 0.5).astype(int)
            cm = confusion_matrix(y_test, pred_rf, labels=[0, 1])
            cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
            st.dataframe(cm_df, use_container_width=True)

            try:
                imp = pd.Series(rfc.feature_importances_, index=class_feats).sort_values(ascending=True)
                fig, ax = plt.subplots(figsize=(7, 4))
                imp.tail(12).plot(kind="barh", ax=ax)
                ax.set_title("Feature Importances (RF)")
                st.pyplot(fig, use_container_width=True)
            except Exception:
                pass

            st.session_state.last_models["classification"] = {"features": class_feats, "metrics": {"LogReg": m_lg, "RF": m_rf}}
            pred_df = pd.DataFrame({"y_true": y_test, "proba_logreg": proba_lg, "proba_rf": proba_rf})
            st.download_button("Download classification predictions", data=to_csv_bytes(pred_df), file_name="classification_predictions.csv", mime="text/csv")

# Hall of Fame Explorer
with tabs[5]:
    st.subheader("Hall of Fame Index Explorer (0-100)")
    st.markdown("Production & Longevity: seasons, games, points, rebounds, assists, avg team win%. Accolade weights: MVP(15), Finals MVP(12), Championships(8), DPOY(5), All-NBA 1st(4), All-NBA Total(2.5), All-Star(2), All-Def 1st(2), All-Def Total(1), ROY(1.5), Scoring Titles(1.5).")
    if career_df.empty:
        st.warning("Career table is empty.")
    else:
        col1, col2 = st.columns(2)
        max_s = int(career_df["seasons"].max()) if "seasons" in career_df.columns and career_df["seasons"].notna().any() else 0
        max_g = int(career_df["games"].max()) if "games" in career_df.columns and career_df["games"].notna().any() else 0
        with col1:
            min_seasons = safe_int_slider("Minimum seasons", 0, max_s, 0, step=1, key="min_seasons")
        with col2:
            g_step = 50 if max_g >= 50 else 1
            min_games = safe_int_slider("Minimum games", 0, max_g, 0, step=g_step, key="min_games")

        filt = career_df.copy()
        if "seasons" in filt.columns:
            filt = filt[filt["seasons"].fillna(0) >= min_seasons]
        if "games" in filt.columns:
            filt = filt[filt["games"].fillna(0) >= min_games]
        if "hof_index" in filt.columns:
            filt = filt.sort_values("hof_index", ascending=False)

        st.write(f"Filtered players: {len(filt):,}")
        if len(filt) > 0:
            top_max = min(200, len(filt))
            top_default = min(50, len(filt))
            top_n = safe_int_slider("Show top N", 10, top_max if top_max >= 10 else 10, top_default if top_default >= 10 else 10, step=10, key="top_n")
            display_cols = [c for c in ["player_name","from_year","to_year","seasons","games","tot_pts","tot_reb","tot_ast"] + accolade_cols_available + ["hof_index"] if c in filt.columns]
            st.dataframe(filt[display_cols].head(top_n), use_container_width=True)

        st.markdown("---")
        st.markdown("Inspect player")
        if "player_name" in career_df.columns:
            players = sorted(career_df["player_name"].dropna().unique())
            if players:
                sel = st.selectbox("Player", players, key="hof_sel")
                row = career_df[career_df["player_name"] == sel]
                if not row.empty:
                    r = row.iloc[0]
                    a, b, c = st.columns(3)
                    with a:
                        st.markdown("Career")
                        for col in ["from_year","to_year","seasons","games"]:
                            if col in r.index and pd.notna(r[col]):
                                v = int(r[col]) if col != "games" else f"{int(r[col]):,}"
                                st.write(f"{col.replace('_', ' ').title()}: {v}")
                    with b:
                        st.markdown("Totals")
                        for col in ["tot_pts","tot_reb","tot_ast"]:
                            if col in r.index and pd.notna(r[col]):
                                st.write(f"{col.replace('tot_','').upper()}: {int(r[col]):,}")
                    with c:
                        st.markdown("Accolades")
                        any_acc = False
                        for col in accolade_cols_available:
                            if col in r.index and pd.notna(r[col]) and r[col] > 0:
                                st.write(f"{col.replace('_',' ').title()}: {int(r[col])}")
                                any_acc = True
                        if not any_acc:
                            st.caption("No accolades recorded.")
                        if "avg_team_win_pct" in r.index and pd.notna(r["avg_team_win_pct"]):
                            st.write(f"Avg Win%: {r['avg_team_win_pct']:.3f}")

                    if "hof_index" in r.index:
                        hof_idx = float(r["hof_index"])
                        st.markdown(f"HoF Index: {hof_idx:.1f} / 100")
                        fig, ax = plt.subplots(figsize=(8, 1.6))
                        ax.barh([0], [hof_idx], height=0.5)
                        ax.set_xlim(0, 100); ax.set_yticks([]); ax.set_xlabel("HoF Index (0-100)")
                        ax.axvline(x=50, color="gray", linestyle="--", alpha=0.3, label="Avg")
                        ax.axvline(x=85, color="blue", linestyle="--", alpha=0.3, label="Strong")
                        ax.axvline(x=95, color="green", linestyle="--", alpha=0.3, label="Elite")
                        ax.legend(loc="upper right", fontsize=8); st.pyplot(fig, use_container_width=True)
                        contrib_cols = [c for c in career_df.columns if c.startswith("hof_c_")]
                        if contrib_cols:
                            contrib = r[contrib_cols].rename(lambda c: c.replace("hof_c_", "")).sort_values(ascending=False)
                            topk_max = max(5, min(20, len(contrib)))
                            topk = safe_int_slider("Show top K contributors", 5, topk_max, min(10, topk_max), step=1, key="topk")
                            cc = contrib.head(topk)
                            fig, ax = plt.subplots(figsize=(8, 4))
                            cc.plot(kind="barh", ax=ax)
                            ax.invert_yaxis(); ax.set_title("Top HoF feature contributions (std-weighted)")
                            st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    st.download_button("Download career table with accolades & HoF Index",
                       data=to_csv_bytes(career_df),
                       file_name="career_with_accolades_hof_index.csv",
                       mime="text/csv")

# About
with tabs[6]:
    render_about_tab()
