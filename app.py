# app.py
# ==================================================
# CMSE 830 FINAL PROJECT
# NBA MULTI-DATASET ANALYTICS + HOF INDEX DASHBOARD
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

# ---------- PAGE ----------
st.set_page_config(page_title="NBA Analytics & Hall of Fame Index", page_icon="🏀", layout="wide")
st.title("🏀 NBA Analytics & Hall of Fame Index Dashboard (Accolade-Driven)")
st.caption("Sources: szymonjwiak/nba-traditional • boonpalipatana/nba-season-records-from-every-year • ryanschubertds/all-nba-aba-players-bio-stats-accolades")

# ---------- CONSTANTS ----------
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
    "all_defensive_first": ["all_defensive_first", "all-defensive_first", "all_def_1st", "all-defense 1st team"],
    "all_defensive_second": ["all_defensive_second", "all-defensive_second", "all_def_2nd", "all-defense 2nd team"],
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

# ---------- HELPERS ----------
def _safe_to_int(x) -> Optional[int]:
    try: return int(x)
    except Exception: return None

def _season_to_start_year(x) -> Optional[int]:
    if pd.isna(x): return None
    s = str(x).strip()
    if "-" in s: s = s.split("-")[0]
    return _safe_to_int(s)

def _normalize_names(series: pd.Series) -> pd.Series:
    return (series.fillna("").astype(str).str.strip()
            .str.replace(r"\.", "", regex=True).str.replace(r"\s+", " ", regex=True).str.title())

def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns: return c
    return None

def safe_int_slider(label: str, min_value: int, max_value: int, value: int, step: int = 1, key: Optional[str] = None) -> int:
    # why: Streamlit slider crashes if min==max or invalid step
    if max_value is None or min_value is None:
        return value
    max_value = int(max_value); min_value = int(min_value); value = int(value)
    if max_value < min_value: max_value = min_value
    if max_value == min_value:
        st.number_input(label, min_value=min_value, max_value=max_value, value=value, step=1, key=key, disabled=True)
        return value
    step = max(1, int(step))
    value = min(max(value, min_value), max_value)
    return st.slider(label, min_value=min_value, max_value=max_value, value=value, step=step, key=key)

# ---------- LOADING (Kaggle + local zip fallback) ----------
def _list_csvs_in_zip(zip_path: str) -> List[str]:
    if not os.path.exists(zip_path): return []
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            return [zi.filename for zi in z.infolist() if zi.filename.lower().endswith(".csv")]
    except Exception:
        return []

def _read_csv_from_zip(zip_path: str, name_hint: Optional[str] = None) -> Optional[pd.DataFrame]:
    if not os.path.exists(zip_path): return None
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            names = _list_csvs_in_zip(zip_path)
            target = None
            if name_hint:
                for n in names:
                    if name_hint.lower() in n.lower():
                        target = n; break
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
    if not files: raise FileNotFoundError(f"No CSV in {dataset_id}")
    files = sorted(files, key=lambda f: os.path.getsize(os.path.join(path, f)), reverse=True)
    return pd.read_csv(os.path.join(path, files[0]))

@st.cache_data(show_spinner=True)
def load_boxscores() -> pd.DataFrame:
    try:
        df = _load_kaggle_csv_any("szymonjwiak/nba-traditional")
    except Exception as e:
        st.warning(f"Kaggle boxscores unavailable ({e}); trying local archive…")
        df = None
        for zp in LOCAL_ZIP_GLOBS:
            df = _read_csv_from_zip(zp, "traditional") or _read_csv_from_zip(zp, "game")
            if df is not None: break
        if df is None: raise
    df.columns = df.columns.str.strip().str.lower()
    rename = {}
    season_col = _first_existing_column(df, ["season", "season_id"])
    player_col = _first_existing_column(df, ["player_name", "player"])
    team_col = _first_existing_column(df, ["team_abbreviation", "team"])
    if season_col: rename[season_col] = "season"
    if player_col: rename[player_col] = "player_name"
    if team_col: rename[team_col] = "team"
    if "game_id" in df.columns: rename["game_id"] = "game_id"
    for stat in ["pts", "reb", "ast", "stl", "blk", "tov", "min"]:
        if stat in df.columns: rename[stat] = stat
    df = df.rename(columns=rename)
    keep = [c for c in ["season", "player_name", "team", "game_id", "pts", "reb", "ast", "stl", "blk"] if c in df.columns]
    df = df[keep].copy()
    if "season" in df.columns:
        df["year"] = df["season"].apply(_season_to_start_year)
        df = df.dropna(subset=["year"])
        df["year"] = df["year"].astype(int)
        df = df[df["year"] >= MIN_YEAR]
    if "player_name" in df.columns: df["player_name"] = _normalize_names(df["player_name"])
    if "team" in df.columns: df["team"] = df["team"].str.upper()
    return df

@st.cache_data(show_spinner=True)
def load_team_records() -> pd.DataFrame:
    def _season_to_year_any(s) -> Optional[int]:
        if pd.isna(s): return None
        s = str(s).strip()
        if "-" in s: return _safe_to_int(s.split("-")[0])
        return _safe_to_int(s)
    try:
        df = _load_kaggle_csv_any("boonpalipatana/nba-season-records-from-every-year")
    except Exception as e:
        st.warning(f"Kaggle team records unavailable ({e}); trying local archive…")
        df = None
        for zp in LOCAL_ZIP_GLOBS:
            df = _read_csv_from_zip(zp, "season") or _read_csv_from_zip(zp, "records")
            if df is not None: break
        if df is None:
            return pd.DataFrame(columns=["year", "team", "win_pct"])
    df.columns = df.columns.str.strip().str.lower()
    year_col = _first_existing_column(df, ["year", "season", "season_end", "season_start"])
    team_col = _first_existing_column(df, ["team", "team_name", "franchise", "club", "name"])
    wins_col = _first_existing_column(df, ["wins", "w"])
    losses_col = _first_existing_column(df, ["losses", "l"])
    winpct_col = _first_existing_column(df, ["win_pct", "win%", "w/l%", "win percent", "win percentage"])
    work = pd.DataFrame()
    if year_col is not None: work["year"] = df[year_col].apply(_season_to_year_any)
    if team_col is not None: work["team"] = df[team_col]
    if wins_col is not None: work["wins"] = pd.to_numeric(df[wins_col], errors="coerce")
    if losses_col is not None: work["losses"] = pd.to_numeric(df[losses_col], errors="coerce")
    if winpct_col is not None: work["win_pct"] = pd.to_numeric(df[winpct_col], errors="coerce")
    if "win_pct" not in work.columns and {"wins", "losses"}.issubset(work.columns):
        denom = (work["wins"] + work["losses"]).replace(0, np.nan)
        work["win_pct"] = (work["wins"] / denom).astype(float)
    if "team" in work.columns: work["team"] = work["team"].astype(str).str.upper().str.strip()
    if "year" in work.columns:
        work = work.dropna(subset=["year"])
        work["year"] = work["year"].astype(int)
        work = work[work["year"] >= MIN_YEAR]
    keep = [c for c in ["year", "team", "win_pct"] if c in work.columns]
    if not keep:
        return pd.DataFrame(columns=["year", "team", "win_pct"])
    return work[keep].dropna(subset=["team"]).drop_duplicates()

@st.cache_data(show_spinner=True)
def load_accolades() -> pd.DataFrame:
    try:
        df = _load_kaggle_csv_any("ryanschubertds/all-nba-aba-players-bio-stats-accolades")
    except Exception as e:
        st.warning(f"Kaggle accolades unavailable ({e}); trying local archive…")
        df = None
        for zp in LOCAL_ZIP_GLOBS:
            df = _read_csv_from_zip(zp, "accolade") or _read_csv_from_zip(zp, "all-nba")
            if df is not None: break
        if df is None:
            return pd.DataFrame(columns=["player_name"])
    df.columns = df.columns.str.strip().str.lower()
    name_col = _first_existing_column(df, ["player", "player_name", "name"])
    if name_col:
        df = df.rename(columns={name_col: "player_name"})
    else:
        return pd.DataFrame(columns=["player_name"])
    df["player_name"] = _normalize_names(df["player_name"])
    standard_cols = {}
    for canonical, candidates in ACC_MAPPING.items():
        for c in candidates:
            if c in df.columns:
                standard_cols[c] = canonical; break
    df = df.rename(columns=standard_cols)
    acc_cols = [c for c in ACC_MAPPING.keys() if c in df.columns]
    if not acc_cols:
        return df[["player_name"]].drop_duplicates()
    for c in acc_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    agg = df.groupby("player_name", as_index=False)[acc_cols].sum()
    if set(["all_nba_first", "all_nba_second", "all_nba_third"]).intersection(agg.columns):
        cols = [c for c in ["all_nba_first", "all_nba_second", "all_nba_third"] if c in agg.columns]
        agg["all_nba_total"] = agg[cols].sum(axis=1)
    if set(["all_defensive_first", "all_defensi_]()
