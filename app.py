# app.py
# ==================================================
# CMSE 830 FINAL PROJECT
# NBA MULTI-DATASET ANALYTICS + HOF INDEX DASHBOARD
# (Datasets used: ONLY the three listed below)
#   1) szymonjwiak/nba-traditional             -> boxscores
#   2) boonpalipatana/nba-season-records-from-every-year -> team win%
#   3) ryanschubertds/all-nba-aba-players-bio-stats-accolades -> awards
# ==================================================
from __future__ import annotations

import io
import os
import zipfile
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Kaggle hub
import kagglehub

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="NBA Analytics & Hall of Fame Index",
    page_icon="🏀",
    layout="wide"
)

st.title("🏀 NBA Analytics & Hall of Fame Index Dashboard (Accolade-Driven)")
st.caption("Sources: szymonjwiak/nba-traditional • boonpalipatana/nba-season-records-from-every-year • ryanschubertds/all-nba-aba-players-bio-stats-accolades")

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------
MIN_YEAR = 2005  # modern-tracking window for boxscores
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
    # Production & Longevity
    "seasons": 1.0,
    "games": 0.6,
    "tot_pts": 0.7,
    "tot_reb": 0.35,
    "tot_ast": 0.45,
    "avg_team_win_pct": 1.2,
    # Accolades
    "mvp": 15.0,
    "finals_mvp": 12.0,
    "championships": 8.0,
    "dpoy": 5.0,
    "all_nba_first": 4.0,
    "all_nba_total": 2.5,
    "all_star": 2.0,
    "all_defensive_first": 2.0,
    "all_defensive_total": 1.0,
    "roy": 1.5,
    "scoring_titles": 1.5,
}

# --------------------------------------------------
# UTILS
# --------------------------------------------------
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
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\.", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.title()
    )

def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

# --------------------------------------------------
# LOADING: KAGGLE & LOCAL ZIP FALLBACK
# --------------------------------------------------
def _list_csvs_in_zip(zip_path: str) -> List[str]:
    if not os.path.exists(zip_path):
        return []
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            return [zi.filename for zi in z.infolist() if zi.filename.lower().endswith(".csv")]
    except Exception:
        return []

def _read_csv_from_zip(zip_path: str, csv_name_contains: Optional[str] = None) -> Optional[pd.DataFrame]:
    if not os.path.exists(zip_path):
        return None
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            names = _list_csvs_in_zip(zip_path)
            target = None
            if csv_name_contains:
                for n in names:
                    if csv_name_contains.lower() in n.lower():
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
    # Prefer largest CSV by size as heuristic unless specific ones are known later.
    files = sorted(files, key=lambda f: os.path.getsize(os.path.join(path, f)), reverse=True)
    return pd.read_csv(os.path.join(path, files[0]))

@st.cache_data(show_spinner=True)
def load_boxscores() -> pd.DataFrame:
    """
    szymonjwiak/nba-traditional → game-level boxscores.
    Fallback: try local zip(s).
    """
    try:
        df = _load_kaggle_csv_any("szymonjwiak/nba-traditional")
    except Exception as e:
        st.warning(f"Kaggle boxscores unavailable ({e}); trying local archive…")
        # Attempt find a CSV with 'traditional' or 'game' hint
        for zp in LOCAL_ZIP_GLOBS:
            df = _read_csv_from_zip(zp, "traditional") or _read_csv_from_zip(zp, "game")
            if df is not None:
                break
        if df is None:
            raise
    df.columns = df.columns.str.strip().str.lower()
    # Standardize columns
    rename = {}
    season_col = _first_existing_column(df, ["season", "season_id"])
    if season_col:
        rename[season_col] = "season"
    player_col = _first_existing_column(df, ["player_name", "player"])
    if player_col:
        rename[player_col] = "player_name"
    team_col = _first_existing_column(df, ["team_abbreviation", "team"])
    if team_col:
        rename[team_col] = "team"
    if "game_id" in df.columns:
        rename["game_id"] = "game_id"
    for stat in ["pts", "reb", "ast", "stl", "blk", "tov", "min"]:
        if stat in df.columns:
            rename[stat] = stat
    df = df.rename(columns=rename)
    # Keep essential columns if present
    keep = [c for c in ["season", "player_name", "team", "game_id", "pts", "reb", "ast", "stl", "blk"] if c in df.columns]
    df = df[keep].copy()
    # Normalize season → year
    if "season" in df.columns:
        df["year"] = df["season"].apply(_season_to_start_year)
        df = df.dropna(subset=["year"])
        df["year"] = df["year"].astype(int)
        df = df[df["year"] >= MIN_YEAR]
    # Normalize names
    if "player_name" in df.columns:
        df["player_name"] = _normalize_names(df["player_name"])
    if "team" in df.columns:
        df["team"] = df["team"].str.upper()
    return df

@st.cache_data(show_spinner=True)
def load_team_records() -> pd.DataFrame:
    """
    boonpalipatana/nba-season-records-from-every-year → team win%.
    Fallback: local zip(s).
    """
    try:
        df = _load_kaggle_csv_any("boonpalipatana/nba-season-records-from-every-year")
    except Exception as e:
        st.warning(f"Kaggle team records unavailable ({e}); trying local archive…")
        df = None
        for zp in LOCAL_ZIP_GLOBS:
            df = _read_csv_from_zip(zp, "season") or _read_csv_from_zip(zp, "records")
            if df is not None:
                break
        if df is None:
            raise
    df.columns = df.columns.str.strip().str.lower()
    # Standardize cols
    rename = {}
    if "season" in df.columns:
        rename["season"] = "year"
    if "team_name" in df.columns:
        rename["team_name"] = "team"
    if "win%" in df.columns:
        rename["win%"] = "win_pct"
    df = df.rename(columns=rename)
    # Coerce numeric
    for c in ["year", "wins", "losses", "win_pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "team" in df.columns:
        df["team"] = df["team"].str.upper()
    if "year" in df.columns:
        df = df[df["year"] >= MIN_YEAR]
    # Keep lightweight
    keep = [c for c in ["year", "team", "win_pct"] if c in df.columns]
    return df[keep].drop_duplicates()

@st.cache_data(show_spinner=True)
def load_accolades() -> pd.DataFrame:
    """
    ryanschubertds/all-nba-aba-players-bio-stats-accolades → player accolades.
    Fallback: local zip(s).
    """
    try:
        df = _load_kaggle_csv_any("ryanschubertds/all-nba-aba-players-bio-stats-accolades")
    except Exception as e:
        st.warning(f"Kaggle accolades unavailable ({e}); trying local archive…")
        df = None
        for zp in LOCAL_ZIP_GLOBS:
            df = _read_csv_from_zip(zp, "accolade") or _read_csv_from_zip(zp, "all-nba")
            if df is not None:
                break
        if df is None:
            raise
    df.columns = df.columns.str.strip().str.lower()
    # Standardize player column
    name_col = _first_existing_column(df, ["player", "player_name", "name"])
    if name_col:
        df = df.rename(columns={name_col: "player_name"})
    else:
        # Without player_name this is unusable
        return pd.DataFrame(columns=["player_name"])
    df["player_name"] = _normalize_names(df["player_name"])

    # Build canonical columns
    standard_cols = {}
    for canonical, candidates in ACC_MAPPING.items():
        for c in candidates:
            if c in df.columns:
                standard_cols[c] = canonical
                break
    df = df.rename(columns=standard_cols)

    # Aggregate per player
    acc_cols = [c for c in ACC_MAPPING.keys() if c in df.columns]
    if not acc_cols:
        return df[["player_name"]].drop_duplicates()

    for c in acc_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    agg = df.groupby("player_name", as_index=False)[acc_cols].sum()

    # Derived totals
    if set(["all_nba_first", "all_nba_second", "all_nba_third"]).intersection(agg.columns):
        cols = [c for c in ["all_nba_first", "all_nba_second", "all_nba_third"] if c in agg.columns]
        agg["all_nba_total"] = agg[cols].sum(axis=1)
    if set(["all_defensive_first", "all_defensive_second"]).intersection(agg.columns):
        cols = [c for c in ["all_defensive_first", "all_defensive_second"] if c in agg.columns]
        agg["all_defensive_total"] = agg[cols].sum(axis=1)
    return agg

# --------------------------------------------------
# BUILD TABLES
# --------------------------------------------------
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

    # Merge team win%
    if not season_df.empty and not teams.empty and set(["year", "team"]).issubset(teams.columns):
        season_df = season_df.merge(teams, on=["year", "team"], how="left")

    # Build career from season_df
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

    # Merge accolades
    if not acc.empty and "player_name" in acc.columns and not career_df.empty:
        career_df = career_df.merge(acc, on="player_name", how="left")
    # Fill NaNs for award columns
    award_cols = [c for c in career_df.columns if c in set(ACC_MAPPING.keys()) | {"all_nba_total", "all_defensive_total"}]
    for c in award_cols:
        career_df[c] = career_df[c].fillna(0)

    # Dtype coercion
    for c in ["from_year", "to_year", "seasons", "games", "tot_pts", "tot_reb", "tot_ast", "tot_stl", "tot_blk"]:
        if c in career_df.columns:
            career_df[c] = pd.to_numeric(career_df[c], errors="coerce")
    if "avg_team_win_pct" in career_df.columns:
        career_df["avg_team_win_pct"] = pd.to_numeric(career_df["avg_team_win_pct"], errors="coerce")

    return season_df, teams, career_df

season_df, team_df, career_df = build_tables()

# --------------------------------------------------
# HOF INDEX
# --------------------------------------------------
def compute_hof_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df = pd.DataFrame(columns=["player_name", "hof_index"])
        return df
    use_cols = []
    # Production/longevity
    use_cols += [c for c in ["seasons", "games", "tot_pts", "tot_reb", "tot_ast", "avg_team_win_pct"] if c in df.columns]
    # Accolades
    use_cols += [c for c in ["mvp", "finals_mvp", "championships", "dpoy",
                             "all_nba_first", "all_nba_total", "all_star",
                             "all_defensive_first", "all_defensive_total",
                             "roy", "scoring_titles"] if c in df.columns]
    if not use_cols:
        out = df.copy()
        out["hof_index"] = 0.0
        return out

    X = df[use_cols].fillna(0.0).astype(float)
    # Weights (only for present columns)
    weights = pd.Series({c: HOF_WEIGHTS.get(c, 0.0) for c in use_cols})
    # Z-score columns to balance scale, avoid div-by-zero
    mu = X.mean()
    sigma = X.std(ddof=0).replace(0, 1.0)
    Z = (X - mu) / sigma
    # Weighted sum
    contrib = Z.mul(weights, axis=1)
    raw = contrib.sum(axis=1)
    # Percentile rank → [0, 100]
    ranks = raw.rank(method="average", pct=True)
    hof_index = (ranks * 100).round(1)

    out = df.copy()
    out["hof_index"] = hof_index
    # Store contributions for explainability
    for c in contrib.columns:
        out[f"hof_c_{c}"] = contrib[c]
    out["hof_raw"] = raw
    return out

career_df = compute_hof_index(career_df)

# Accolade availability for UI
ALL_POSSIBLE_ACCS = [
    "mvp", "finals_mvp", "championships", "dpoy",
    "all_nba_first", "all_nba_second", "all_nba_third", "all_nba_total",
    "all_star", "all_defensive_first", "all_defensive_second", "all_defensive_total",
    "roy", "scoring_titles"
]
accolade_cols_available = [c for c in ALL_POSSIBLE_ACCS if c in career_df.columns]

# --------------------------------------------------
# TABS
# --------------------------------------------------
tabs = st.tabs([
    "Overview",
    "EDA (Season & Team)",
    "Player Explorer",
    "Player Comparison",
    "Team Trends",
    "Hall of Fame Explorer"
])

# TAB 1: OVERVIEW
with tabs[0]:
    st.subheader("Dataset overview")
    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.metric("Total Players", f"{len(career_df):,}")
    with col_b:
        st.metric("Players (Season Data)", f"{season_df['player_name'].nunique():,}" if "player_name" in season_df.columns else "0")
    with col_c:
        st.metric("Player-Seasons", f"{len(season_df):,}")
    with col_d:
        if "year" in season_df.columns and not season_df.empty:
            st.metric("Year Range", f"{int(season_df['year'].min())}-{int(season_df['year'].max())}")
        else:
            st.metric("Year Range", "N/A")

    st.markdown("### 🏆 Accolades Data Status")
    if accolade_cols_available:
        display = {
            "mvp": "MVP", "finals_mvp": "Finals MVP", "championships": "Championships",
            "dpoy": "DPOY", "all_nba_first": "All-NBA 1st", "all_nba_second": "All-NBA 2nd",
            "all_nba_third": "All-NBA 3rd", "all_nba_total": "All-NBA Total", "all_star": "All-Star",
            "all_defensive_first": "All-Defense 1st", "all_defensive_second": "All-Defense 2nd",
            "all_defensive_total": "All-Defense Total", "roy": "ROY", "scoring_titles": "Scoring Titles"
        }
        st.success("Included: " + ", ".join(display.get(c, c) for c in accolade_cols_available))
    else:
        st.warning("No accolades columns detected.")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Season-level (sample)**")
        st.dataframe(season_df.head(20), use_container_width=True)
    with c2:
        st.markdown("**Career-level (sample)**")
        show_cols = [c for c in ["player_name", "from_year", "to_year", "seasons", "games", "tot_pts", "tot_reb", "tot_ast", "avg_team_win_pct", "hof_index"] if c in career_df.columns]
        st.dataframe(career_df[show_cols].head(20), use_container_width=True)

# TAB 2: EDA
with tabs[1]:
    st.subheader("Season & Team EDA")
    if not season_df.empty:
        numeric_cols = season_df.select_dtypes(include=np.number).columns.tolist()
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Correlation heatmap**")
            if len(numeric_cols) >= 2:
                method = st.selectbox("Correlation method", ["pearson", "spearman"], index=0)
                corr = season_df[numeric_cols].corr(method=method)
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
                ax.set_title(f"Correlation ({method})")
                st.pyplot(fig, use_container_width=True)
        with colB:
            st.markdown("**Distribution**")
            if numeric_cols:
                stat = st.selectbox("Select stat", numeric_cols, index=0)
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.histplot(season_df[stat].dropna(), kde=True, bins=30, ax=ax)
                ax.set_title(f"Distribution of {stat}")
                st.pyplot(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### Scatter: Stat vs Team Win%")
        if "win_pct" in season_df.columns:
            stat_choice = st.selectbox("Choose stat", [c for c in ["pts", "reb", "ast", "stl", "blk", "games"] if c in season_df.columns])
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=season_df, x=stat_choice, y="win_pct", alpha=0.35, ax=ax)
            ax.set_title(f"{stat_choice} vs Team Win% (player-season)")
            st.pyplot(fig, use_container_width=True)

# TAB 3: PLAYER EXPLORER
with tabs[2]:
    st.subheader("Player explorer (season-level)")
    if "player_name" in season_df.columns and not season_df.empty:
        players = sorted(season_df["player_name"].dropna().unique())
        sel_player = st.selectbox("Select a player", players)
        pdf = season_df[season_df["player_name"] == sel_player].sort_values("year")
        st.dataframe(pdf, use_container_width=True)
        stat_choices = [c for c in ["pts", "reb", "ast", "stl", "blk"] if c in pdf.columns]
        if stat_choices:
            stat_to_plot = st.selectbox("Plot stat over time", stat_choices, index=0)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(pdf["year"], pdf[stat_to_plot], marker="o")
            ax.set_xlabel("Year")
            ax.set_ylabel(stat_to_plot.upper())
            ax.set_title(f"{sel_player} – {stat_to_plot.upper()} over seasons")
            st.pyplot(fig, use_container_width=True)

        # Rolling mean
        if "pts" in pdf.columns:
            window = st.slider("Rolling window (seasons)", 1, min(10, len(pdf)), 3)
            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(pdf["year"], pdf["pts"], alpha=0.4)
            ax.plot(pdf["year"], pdf["pts"].rolling(window, min_periods=1).mean(), linewidth=2)
            ax.set_title(f"{sel_player} – Points (rolling {window})")
            st.pyplot(fig, use_container_width=True)

# TAB 4: PLAYER COMPARISON
with tabs[3]:
    st.subheader("Player Comparison")
    if "player_name" in career_df.columns and len(career_df) >= 2:
        plist = sorted(career_df["player_name"].unique())
        col1, col2 = st.columns(2)
        with col1:
            p1 = st.selectbox("Select Player 1", plist, key="cmp_p1")
        with col2:
            p2 = st.selectbox("Select Player 2", plist, index=1 if len(plist) > 1 else 0, key="cmp_p2")
        row1 = career_df[career_df["player_name"] == p1].iloc[0]
        row2 = career_df[career_df["player_name"] == p2].iloc[0]

        # Table
        st.markdown("### Career Statistics Comparison")
        comp_stats = ["seasons", "games", "tot_pts", "tot_reb", "tot_ast", "avg_team_win_pct", "hof_index"]
        comp_stats += [c for c in accolade_cols_available if c in row1.index and c in row2.index]
        rows = []
        for s in comp_stats:
            v1 = row1.get(s, np.nan)
            v2 = row2.get(s, np.nan)
            if s == "avg_team_win_pct" and pd.notna(v1) and pd.notna(v2):
                f1 = f"{float(v1):.3f}"
                f2 = f"{float(v2):.3f}"
            elif s == "hof_index" and pd.notna(v1) and pd.notna(v2):
                f1 = f"{float(v1):.1f}"
                f2 = f"{float(v2):.1f}"
            else:
                f1 = f"{int(v1):,}" if pd.notna(v1) else "–"
                f2 = f"{int(v2):,}" if pd.notna(v2) else "–"
            rows.append({"Statistic": s.replace("tot_", "Total ").replace("_", " ").title(), p1: f1, p2: f2})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Contribution bars
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
        _plot_contrib(row1, f"{p1} – Top HOF contributors")
        _plot_contrib(row2, f"{p2} – Top HOF contributors")

# TAB 5: TEAM TRENDS
with tabs[4]:
    st.subheader("Team Trends")
    if not team_df.empty and set(["team", "year", "win_pct"]).issubset(team_df.columns):
        team_list = sorted(team_df["team"].dropna().unique())
        sel_team = st.selectbox("Team", team_list)
        tdf = team_df[team_df["team"] == sel_team].sort_values("year")
        st.dataframe(tdf, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(tdf["year"], tdf["win_pct"], marker="o")
        ax.set_xlabel("Year")
        ax.set_ylabel("Win%")
        ax.set_title(f"{sel_team} – Win% over time")
        st.pyplot(fig, use_container_width=True)

        if not season_df.empty:
            ssub = season_df[season_df["team"] == sel_team]
            agg_cols = {c: "sum" for c in ["pts", "reb", "ast"] if c in ssub.columns}
            if "games" in ssub.columns:
                agg_cols["games"] = "sum"
            if agg_cols:
                team_season_stats = ssub.groupby("year").agg(agg_cols).reset_index()
                st.markdown("### Team season totals")
                st.dataframe(team_season_stats, use_container_width=True)
                if "pts" in team_season_stats.columns:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(team_season_stats["year"], team_season_stats["pts"], marker="o")
                    ax.set_xlabel("Year")
                    ax.set_ylabel("Total Points")
                    ax.set_title(f"{sel_team} – Total points per season")
                    st.pyplot(fig, use_container_width=True)

# TAB 6: HALL OF FAME EXPLORER
with tabs[5]:
    st.subheader("Hall of Fame Index Explorer (0–100)")

    st.markdown(
        """
**Production & Longevity:** seasons, games, points, rebounds, assists, avg team win%.  
**Accolade weights:** MVP(15), Finals MVP(12), Championships(8), DPOY(5), All-NBA 1st(4), All-NBA Total(2.5), All-Star(2), All-Def 1st(2), All-Def Total(1), ROY(1.5), Scoring Titles(1.5).
        """
    )
    if career_df.empty:
        st.warning("Career table is empty.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            max_s = int(career_df["seasons"].max()) if "seasons" in career_df.columns and career_df["seasons"].notna().any() else 0
            min_seasons = st.slider("Minimum seasons", 0, max_s if max_s > 0 else 0, 0)
        with col2:
            max_g = int(career_df["games"].max()) if "games" in career_df.columns and career_df["games"].notna().any() else 0
            min_games = st.slider("Minimum games", 0, max_g if max_g > 0 else 0, 0, step=50)

        filt = career_df.copy()
        if "seasons" in filt.columns:
            filt = filt[filt["seasons"].fillna(0) >= min_seasons]
        if "games" in filt.columns:
            filt = filt[filt["games"].fillna(0) >= min_games]

        filt = filt.sort_values("hof_index", ascending=False)
        st.write(f"**Filtered players:** {len(filt):,}")

        top_n = st.slider("Show top N", 10, min(200, len(filt) if len(filt) > 0 else 10), min(50, len(filt) if len(filt) > 0 else 10), 10)
        display_cols = [c for c in ["player_name", "from_year", "to_year", "seasons", "games",
                                    "tot_pts", "tot_reb", "tot_ast"] + accolade_cols_available + ["hof_index"]
                        if c in filt.columns]
        if display_cols and not filt.empty:
            st.dataframe(filt[display_cols].head(top_n), use_container_width=True)

        st.markdown("---")
        st.markdown("### Inspect player")
        players = sorted(career_df["player_name"].dropna().unique())
        if players:
            sel = st.selectbox("Player", players, key="hof_sel")
            row = career_df[career_df["player_name"] == sel]
            if not row.empty:
                r = row.iloc[0]
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Career**")
                    for col in ["from_year", "to_year", "seasons", "games"]:
                        if col in r.index and pd.notna(r[col]):
                            v = int(r[col]) if col != "games" else f"{int(r[col]):,}"
                            st.write(f"{col.replace('_', ' ').title()}: {v}")
                with c2:
                    st.markdown("**Totals**")
                    for col in ["tot_pts", "tot_reb", "tot_ast"]:
                        if col in r.index and pd.notna(r[col]):
                            st.write(f"{col.replace('tot_', '').upper()}: {int(r[col]):,}")
                with c3:
                    st.markdown("**Accolades**")
                    any_acc = False
                    for col in accolade_cols_available:
                        if col in r.index and pd.notna(r[col]) and r[col] > 0:
                            st.write(f"{col.replace('_', ' ').title()}: {int(r[col])}")
                            any_acc = True
                    if not any_acc:
                        st.caption("No accolades recorded.")

                if "hof_index" in r.index:
                    hof_idx = float(r["hof_index"])
                    st.markdown(f"### HoF Index: **{hof_idx:.1f} / 100**")
                    if hof_idx >= 95:
                        verdict, color = "🏆 Elite / Inner-circle HoF", "green"
                    elif hof_idx >= 85:
                        verdict, color = "⭐ Strong HoF candidate", "blue"
                    elif hof_idx >= 70:
                        verdict, color = "🎯 Borderline HoF", "orange"
                    elif hof_idx >= 50:
                        verdict, color = "✅ Solid career", "gray"
                    else:
                        verdict, color = "📊 Role player", "lightgray"
                    st.markdown(f"**{verdict}**")
                    fig, ax = plt.subplots(figsize=(8, 1.6))
                    ax.barh([0], [hof_idx], height=0.5, color=color)  # why: encode score perception quickly
                    ax.set_xlim(0, 100)
                    ax.set_yticks([])
                    ax.set_xlabel("HoF Index (0–100)")
                    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.3, label='Avg')
                    ax.axvline(x=85, color='blue', linestyle='--', alpha=0.3, label='Strong')
                    ax.axvline(x=95, color='green', linestyle='--', alpha=0.3, label='Elite')
                    ax.legend(loc='upper right', fontsize=8)
                    st.pyplot(fig, use_container_width=True)

                    # Contributions
                    contrib_cols = [c for c in career_df.columns if c.startswith("hof_c_")]
                    if contrib_cols:
                        contrib = r[contrib_cols].rename(lambda c: c.replace("hof_c_", ""))
                        contrib = contrib.sort_values(ascending=False)
                        topk = st.slider("Show top K contributors", 5, min(20, len(contrib)), 10)
                        cc = contrib.head(topk)
                        fig, ax = plt.subplots(figsize=(8, 4))
                        cc.plot(kind="barh", ax=ax)
                        ax.invert_yaxis()
                        ax.set_title("Top HoF feature contributions (std-weighted)")
                        st.pyplot(fig, use_container_width=True)

        # Scatter: HoF vs accolades
        st.markdown("---")
        st.markdown("### HoF vs Accolades")
        acc_for_scatter = [c for c in ["mvp", "finals_mvp", "championships", "all_star", "all_nba_total"] if c in career_df.columns]
        if acc_for_scatter:
            xacc = st.selectbox("Accolade", acc_for_scatter)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.stripplot(data=career_df, x=xacc, y="hof_index", alpha=0.5, ax=ax)
            ax.set_title(f"HoF Index vs {xacc}")
            st.pyplot(fig, use_container_width=True)

    # Download
    st.markdown("---")
    def to_csv_bytes(df: pd.DataFrame) -> bytes:
        return df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download career table with accolades & HoF Index",
        data=to_csv_bytes(career_df),
        file_name="career_with_accolades_hof_index.csv",
        mime="text/csv"
    )
