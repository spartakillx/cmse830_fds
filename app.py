# ==================================================
# FINAL PROJECT: NBA MULTI-SOURCE ANALYTICS + HoF MODEL
# Seasons: modern era (2004-05 and later)
# Data sources (all via kagglehub):
#   1) drgilermo/nba-players-stats  -> Seasons_Stats (1950+)
#   2) szymonjwiak/nba-traditional  -> Game boxscores (1997+)
#   3) boonpalipatana/nba-season-records-from-every-year -> Season W/L records
# ==================================================
import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns

import kagglehub  # pip install kagglehub

# -------------------------
# GLOBAL SETTINGS
# -------------------------
MODERN_START_YEAR = 2005        # 2004-05 season ends in 2005
RECENT_YEARS_WINDOW = 10        # last 10 seasons for some views

st.set_page_config(
    page_title="NBA Analytics & Hall of Fame Model",
    page_icon="🏀",
    layout="wide"
)

sns.set_style("whitegrid")


# ==================================================
# 1. HELPER: FIND CSV BY REQUIRED COLUMNS
# ==================================================
def find_csv_with_columns(folder: str, required_cols: list, nrows: int = 100):
    """
    Search all CSVs in a folder and return the first path whose columns
    (after normalization) contain all required_cols.
    """
    csv_files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
    for fname in csv_files:
        fpath = os.path.join(folder, fname)
        try:
            tmp = pd.read_csv(fpath, nrows=nrows)
        except Exception:
            continue
        cols = (
            tmp.columns
            .str.strip()
            .str.replace(r"\s+", "_", regex=True)
            .str.replace("%", "pct")
            .str.lower()
        )
        if all(c in cols for c in required_cols):
            return fpath
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("%", "pct")
        .str.lower()
    )
    return df


# ==================================================
# 2. DATA LOADING (3 DATASETS) + BASIC CLEANING
# ==================================================
@st.cache_data(show_spinner="Downloading & loading Seasons_Stats (players)…")
def load_players_season_stats() -> pd.DataFrame:
    path = kagglehub.dataset_download("drgilermo/nba-players-stats")
    # We expect "Seasons_Stats.csv"
    seasons_path = find_csv_with_columns(
        path, required_cols=["year", "player", "pts", "trb", "ast"]
    )
    if seasons_path is None:
        raise FileNotFoundError(
            "Could not find Seasons_Stats-like CSV in drgilermo/nba-players-stats dataset."
        )

    df = pd.read_csv(seasons_path)
    df = normalize_columns(df)

    # Keep modern era only
    if "year" in df.columns:
        df = df[df["year"] >= MODERN_START_YEAR].copy()

    # Coerce common numeric columns
    numeric_like = [
        "g", "mp", "pts", "trb", "ast", "stl", "blk",
        "fg", "fga", "fgpct", "x3p", "x3pa", "x3ppct", "ft", "fta", "ftpct",
        "ws", "ws_48", "vorp", "per", "ts_pct"
    ]
    for col in numeric_like:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # If TS% or 3P% not present, derive some advanced metrics
    if {"pts", "fga", "fta"}.issubset(df.columns):
        denom = 2 * (df["fga"].fillna(0) + 0.44 * df["fta"].fillna(0))
        denom = denom.replace(0, np.nan)
        df["ts_pct"] = df["pts"] / denom

    if {"fg", "x3p", "fga"}.issubset(df.columns):
        denom2 = df["fga"].replace(0, np.nan)
        df["efg_pct"] = (df["fg"] + 0.5 * df["x3p"]) / denom2

    # Clean strings
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    return df


@st.cache_data(show_spinner="Downloading & loading NBA traditional boxscores…")
def load_traditional_boxscores() -> pd.DataFrame:
    path = kagglehub.dataset_download("szymonjwiak/nba-traditional")

    # We want PLAYER-level boxscores
    player_box_path = find_csv_with_columns(
        path,
        required_cols=["player_name", "team_abbreviation", "pts"],
    )
    if player_box_path is None:
        # Fall back to any CSV with 'player' & 'pts'
        player_box_path = find_csv_with_columns(
            path,
            required_cols=["player", "pts"],
        )
    if player_box_path is None:
        raise FileNotFoundError(
            "Could not find player traditional boxscore CSV in szymonjwiak/nba-traditional dataset."
        )

    df = pd.read_csv(player_box_path)
    df = normalize_columns(df)

    # Try to infer season/year
    # Many NBA.com style datasets have 'season' or 'season_id' or 'game_date'
    season_col = None
    for cand in ["season", "season_id", "year"]:
        if cand in df.columns:
            season_col = cand
            break

    if season_col is None and "game_date" in df.columns:
        df["season_year"] = pd.to_datetime(df["game_date"], errors="coerce").dt.year
        season_col = "season_year"

    if season_col is None:
        # As a fallback, keep everything
        df["season_year"] = np.nan
        season_col = "season_year"

    # Normalize season to numeric year-like
    if season_col == "season_id":
        # e.g., 22014 -> 2014-15 season => we map to 2015
        df["season_year"] = (
            df["season_id"]
            .astype(str)
            .str[-2:]
            .astype(int, errors="ignore") + 2000
        )
    elif season_col == "season":
        # e.g., "2014-15" -> 2015
        df["season_year"] = (
            df["season"]
            .astype(str)
            .str.extract(r"(\d{4})")[0]
            .astype(int, errors="ignore") + 1
        )
    else:
        df["season_year"] = pd.to_numeric(df[season_col], errors="coerce")

    # Filter to modern era
    df = df[df["season_year"] >= MODERN_START_YEAR].copy()

    # Coerce key numeric columns
    for col in ["pts", "reb", "ast", "stl", "blk", "tov", "min"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # String cleanup
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    return df


@st.cache_data(show_spinner="Downloading & loading season records…")
def load_season_records() -> pd.DataFrame:
    path = kagglehub.dataset_download("boonpalipatana/nba-season-records-from-every-year")

    # Find a file with season/team/wins/losses
    records_path = find_csv_with_columns(
        path,
        required_cols=["team", "w", "l"],
    )
    if records_path is None:
        # Try other combos
        records_path = find_csv_with_columns(
            path,
            required_cols=["team", "wins", "losses"],
        )
    if records_path is None:
        raise FileNotFoundError(
            "Could not find season records CSV in boonpalipatana/nba-season-records-from-every-year dataset."
        )

    df = pd.read_csv(records_path)
    df = normalize_columns(df)

    # Try to unify columns
    rename_map = {}
    if "season" in df.columns and "year" not in df.columns:
        # 'season' like "2014-15" -> end year 2015
        df["year"] = (
            df["season"]
            .astype(str)
            .str.extract(r"(\d{4})")[0]
            .astype(int, errors="ignore") + 1
        )
    if "wins" in df.columns:
        rename_map["wins"] = "w"
    if "losses" in df.columns:
        rename_map["losses"] = "l"

    df = df.rename(columns=rename_map)

    # Only keep rows with year and W/L
    if "year" not in df.columns:
        # Try direct 'season_end' as year
        if "season_end" in df.columns:
            df["year"] = pd.to_numeric(df["season_end"], errors="coerce")
        else:
            df["year"] = np.nan

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["w"] = pd.to_numeric(df.get("w", np.nan), errors="coerce")
    df["l"] = pd.to_numeric(df.get("l", np.nan), errors="coerce")

    df = df.dropna(subset=["year", "team"]).copy()
    df = df[df["year"] >= MODERN_START_YEAR]

    # Win%
    df["win_pct"] = df["w"] / (df["w"] + df["l"])

    # Very rough 'playoff' flag if column exists
    playoff_flag = np.zeros(len(df))
    for cand in ["playoffs", "playoff", "note"]:
        if cand in df.columns:
            playoff_flag = df[cand].astype(str).str.contains("playoff", case=False, na=False)
            break
    df["playoff_flag"] = playoff_flag.astype(int)

    # Championship flag if any column suggests it
    champ_flag = np.zeros(len(df))
    for cand in ["champion", "championship", "notes"]:
        if cand in df.columns:
            champ_flag = df[cand].astype(str).str.contains("champ", case=False, na=False)
            break
    df["champ_flag"] = champ_flag.astype(int)

    return df


# ==================================================
# 3. DERIVED TABLES: PLAYER CAREERS + HoF SCORE
# ==================================================
@st.cache_data(show_spinner="Building player career table & HoF model…")
def build_career_and_hof(players_season: pd.DataFrame,
                         season_records: pd.DataFrame) -> pd.DataFrame:
    df = players_season.copy()

    # Align team key
    for cand in ["tm", "team"]:
        if cand in df.columns:
            df.rename(columns={cand: "team"}, inplace=True)
            break

    # Join team win% for each season
    records = season_records.copy()
    # unify team col
    if "team" not in records.columns:
        # best effort: pick 'franchise' or similar
        for cand in ["franchise", "tm"]:
            if cand in records.columns:
                records.rename(columns={cand: "team"}, inplace=True)
                break

    join_cols = [c for c in ["year", "team"] if c in df.columns and c in records.columns]
    if join_cols:
        df = df.merge(
            records[["year", "team", "win_pct", "playoff_flag", "champ_flag"]],
            on=join_cols,
            how="left"
        )
    else:
        df["win_pct"] = np.nan
        df["playoff_flag"] = np.nan
        df["champ_flag"] = np.nan

    # Aggregate career-level stats per player
    # We use totals + per-game + team impact
    group = df.groupby("player", dropna=True)

    career = pd.DataFrame({
        "seasons_played": group["year"].nunique(),
        "teams_played_for": group["team"].nunique(),
        "games_played": group["g"].sum(min_count=1) if "g" in df.columns else group.size(),
        "total_pts": group["pts"].sum(min_count=1) if "pts" in df.columns else np.nan,
        "total_trb": group["trb"].sum(min_count=1) if "trb" in df.columns else np.nan,
        "total_ast": group["ast"].sum(min_count=1) if "ast" in df.columns else np.nan,
        "total_ws": group["ws"].sum(min_count=1) if "ws" in df.columns else np.nan,
        "total_vorp": group["vorp"].sum(min_count=1) if "vorp" in df.columns else np.nan,
        "avg_per": group["per"].mean() if "per" in df.columns else np.nan,
        "avg_ts_pct": group["ts_pct"].mean() if "ts_pct" in df.columns else np.nan,
        "avg_efg_pct": group["efg_pct"].mean() if "efg_pct" in df.columns else np.nan,
        "avg_team_win_pct": group["win_pct"].mean(),
        "title_seasons": group["champ_flag"].sum(min_count=1),
        "playoff_seasons": group["playoff_flag"].sum(min_count=1),
        "first_year": group["year"].min(),
        "last_year": group["year"].max(),
    })

    # Per-game rates
    with np.errstate(divide="ignore", invalid="ignore"):
        career["pts_per_g"] = career["total_pts"] / career["games_played"]
        career["trb_per_g"] = career["total_trb"] / career["games_played"]
        career["ast_per_g"] = career["total_ast"] / career["games_played"]

    # Fill games_played=0 -> NaN per-game
    career.loc[career["games_played"] == 0, ["pts_per_g", "trb_per_g", "ast_per_g"]] = np.nan

    # ---- HoF scoring model (heuristic but advanced) ----
    # We compute z-scores for key features and make a weighted index,
    # then pass through logistic transform to get [0,1] "probability".
    def zscore(series):
        return (series - series.mean()) / (series.std(ddof=0) + 1e-9)

    # Choose features that matter most to Hall-of-Fame narrative:
    #   - career volume (total_ws, total_vorp, total_pts)
    #   - per-game impact (pts_per_g, ast_per_g, trb_per_g, avg_per)
    #   - team success (avg_team_win_pct, title_seasons, playoff_seasons)
    for col in [
        "total_ws", "total_vorp", "total_pts",
        "pts_per_g", "ast_per_g", "trb_per_g", "avg_per",
        "avg_team_win_pct", "title_seasons", "playoff_seasons"
    ]:
        if col in career.columns:
            career[col + "_z"] = zscore(career[col].fillna(0))
        else:
            career[col + "_z"] = 0.0

    # Weighted sum -> HoF score
    hof_score = (
        0.25 * career["total_ws_z"] +
        0.20 * career["total_vorp_z"] +
        0.15 * career["pts_per_g_z"] +
        0.08 * career["ast_per_g_z"] +
        0.08 * career["trb_per_g_z"] +
        0.10 * career["avg_per_z"] +
        0.07 * career["avg_team_win_pct_z"] +
        0.04 * career["title_seasons_z"] +
        0.03 * career["playoff_seasons_z"]
    )

    career["hof_score_raw"] = hof_score

    # Logistic transform → (0,1) probability-like measure
    # We rescale hof_score to be a bit sharper (tune factor)
    scale_factor = 0.9
    hof_prob = 1 / (1 + np.exp(-scale_factor * hof_score))
    career["hof_prob"] = hof_prob.clip(0, 1)

    # Categorize
    bins = [0, 0.25, 0.5, 0.75, 1.01]
    labels = ["Unlikely", "On the bubble", "Strong case", "Virtual lock"]
    career["hof_tier"] = pd.cut(career["hof_prob"], bins=bins, labels=labels, include_lowest=True)

    # Sort by HoF probability
    career = career.sort_values("hof_prob", ascending=False).reset_index()
    career.rename(columns={"player": "player_name"}, inplace=True)

    return career


# ==================================================
# 4. LOAD DATA ONCE
# ==================================================
players_season_df = load_players_season_stats()
traditional_df = load_traditional_boxscores()
season_records_df = load_season_records()
career_df = build_career_and_hof(players_season_df, season_records_df)


# ==================================================
# 5. APP TABS
# ==================================================
tabs = st.tabs([
    "Overview",
    "EDA",
    "Player Comparison",
    "Team / Franchise History",
    "HoF Model",
    "Missing Data"
])

# -------------------------
# TAB 0: OVERVIEW
# -------------------------
with tabs[0]:
    st.title("🏀 NBA Analytics & Hall of Fame Probability Dashboard")
    st.markdown("""
This app combines **three NBA datasets** (player seasons, game boxscores, and team season records)  
to analyze **modern era (2004–05 onward)** performance and estimate a **Hall of Fame probability** for each player.

**Data sources (via Kaggle):**
- `drgilermo/nba-players-stats` – Seasons_Stats (player season-level, 1950+)
- `szymonjwiak/nba-traditional` – game-level boxscores (players & teams, 1997–present)
- `boonpalipatana/nba-season-records-from-every-year` – team season W/L + playoff / title indicators
""")

    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.subheader("Sample of player season-level data (modern era)")
        st.dataframe(players_season_df.head(20))
        st.write(f"**Seasons_Stats shape (modern era):** {players_season_df.shape[0]} rows × {players_season_df.shape[1]} columns")

    with c2:
        st.subheader("Missingness summary (season stats)")
        miss = players_season_df.isna().mean().sort_values(ascending=False)
        st.dataframe(miss.to_frame("missing_frac"))

    st.markdown("### Download cleaned career-level table")
    def to_csv_bytes(df_):
        return df_.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download career table with HoF probabilities",
        data=to_csv_bytes(career_df),
        file_name="nba_career_hof_model.csv",
        mime="text/csv"
    )


# -------------------------
# TAB 1: EDA
# -------------------------
with tabs[1]:
    st.header("📊 Exploratory Data Analysis (Season-Level)")

    df = players_season_df.copy()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    colA, colB = st.columns(2)

    # A. Correlation heatmap
    with colA:
        st.subheader("Correlation Heatmap")
        if len(num_cols) >= 2:
            method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"], index=0)
            corr = df[num_cols].corr(method=method)
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
            ax.set_title(f"Season-level correlation ({method.title()})")
            st.pyplot(fig)
        else:
            st.info("Need at least 2 numeric columns for a correlation heatmap.")

    # B. Distribution (stat + optional position filter)
    with colB:
        st.subheader("Stat Distribution")
        stat = st.selectbox("Select numeric stat", num_cols, index=min(3, len(num_cols)-1))

        pos_col = None
        for cand in ["pos", "position"]:
            if cand in df.columns:
                pos_col = cand
                break

        if pos_col:
            pos_options = ["All"] + sorted(df[pos_col].dropna().unique())
            pos_filter = st.selectbox("Filter by position (optional)", pos_options)
            plot_df = df if pos_filter == "All" else df[df[pos_col] == pos_filter]
        else:
            plot_df = df

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(plot_df[stat].dropna(), kde=True, bins=30, ax=ax)
        ax.set_title(f"Distribution of {stat}" + ("" if pos_col is None else f" ({pos_filter})"))
        st.pyplot(fig)

    st.markdown("---")

    # C. Boxplot: stat by team (season-level)
    st.subheader("Stat by Team (Boxplot, Season-Level)")
    team_col = None
    for cand in ["team", "tm"]:
        if cand in df.columns:
            team_col = cand
            break

    if team_col and num_cols:
        stat2 = st.selectbox("Boxplot stat", num_cols, index=0)
        # top teams by average of stat2
        top_teams = (
            df.groupby(team_col)[stat2]
            .mean(numeric_only=True)
            .sort_values(ascending=False)
            .head(12)
            .index
        )
        bp_df = df[df[team_col].isin(top_teams)]
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.boxplot(data=bp_df, x=team_col, y=stat2, ax=ax)
        ax.set_title(f"{stat2} distribution for top 12 {team_col} by mean {stat2}")
        ax.tick_params(axis="x", rotation=0)
        st.pyplot(fig)
    else:
        st.info("No team column found for boxplot.")

    # D. Time trend of a stat (league-wide)
    if "year" in df.columns and num_cols:
        st.subheader("League-wide Trend Over Time")
        stat3 = st.selectbox("Time trend stat", num_cols, index=min(1, len(num_cols)-1))
        tl = (
            df[["year", stat3]]
            .dropna()
            .groupby("year", as_index=False)
            .mean(numeric_only=True)
            .sort_values("year")
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(tl["year"], tl[stat3], marker="o")
        ax.set_xlabel("Season end year")
        ax.set_ylabel(stat3)
        ax.set_title(f"Average {stat3} over seasons (league-wide)")
        st.pyplot(fig)


# -------------------------
# TAB 2: PLAYER COMPARISON
# -------------------------
with tabs[2]:
    st.header("🆚 Head-to-Head Player Comparison (Season Aggregates)")

    players = sorted(players_season_df["player"].dropna().unique())
    if len(players) < 2:
        st.info("Not enough distinct players in dataset.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            p1 = st.selectbox("Player 1", players, key="cmp_p1")
        with col2:
            p2 = st.selectbox("Player 2", players, key="cmp_p2")

        # Choose metrics to compare
        default_stats = [c for c in ["pts", "trb", "ast", "stl", "blk", "ws", "vorp", "per"] if c in players_season_df.columns]
        metrics = st.multiselect("Metrics to compare (season mean across modern era)", default_stats, default=default_stats)

        sub = players_season_df[players_season_df["player"].isin([p1, p2])]
        if not sub.empty and metrics:
            grp = sub.groupby("player")[metrics].mean(numeric_only=True)

            # Bar chart
            fig, ax = plt.subplots(figsize=(8, 4))
            grp.T.plot(kind="bar", ax=ax)
            ax.set_title(f"{p1} vs {p2} — average season stats (modern era)")
            ax.set_ylabel("Average per season")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            st.pyplot(fig)

            # Radar (normalized)
            if grp.shape[0] == 2:
                def radar_plot(data):
                    cats = list(data.columns)
                    # Normalize columns
                    norm = (data - data.min()) / (data.max() - data.min() + 1e-9)
                    values1 = norm.iloc[0].fillna(0).values
                    values2 = norm.iloc[1].fillna(0).values

                    # close the polygon
                    values1 = np.concatenate([values1, values1[:1]])
                    values2 = np.concatenate([values2, values2[:1]])

                    angles = np.linspace(0, 2*np.pi, len(cats), endpoint=False)
                    angles = np.concatenate([angles, [angles[0]]])

                    fig2 = plt.figure(figsize=(6, 6))
                    ax2 = plt.subplot(111, polar=True)
                    ax2.plot(angles, values1, label=p1)
                    ax2.fill(angles, values1, alpha=0.1)
                    ax2.plot(angles, values2, label=p2)
                    ax2.fill(angles, values2, alpha=0.1)
                    ax2.set_xticks(angles[:-1])
                    ax2.set_xticklabels(cats)
                    ax2.set_title("Radar (normalized season averages)")
                    ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
                    return fig2

                st.pyplot(radar_plot(grp))
        else:
            st.info("Select at least one metric to compare.")


# -------------------------
# TAB 3: TEAM / FRANCHISE HISTORY (LAST 10+ YEARS)
# -------------------------
with tabs[3]:
    st.header("🏟️ Team / Franchise History")

    # Use traditional_df (game-level) + season_records_df (W/L)
    #  -> last RECENT_YEARS_WINDOW seasons based on 'season_year'
    df_g = traditional_df.copy()
    if "season_year" not in df_g.columns:
        st.warning("Could not infer season_year from traditional boxscores dataset.")
    else:
        recent_year_cut = df_g["season_year"].max() - RECENT_YEARS_WINDOW + 1
        df_recent = df_g[df_g["season_year"] >= recent_year_cut].copy()

        team_col = None
        for cand in ["team_abbreviation", "team", "tm"]:
            if cand in df_recent.columns:
                team_col = cand
                break

        if team_col is None:
            st.warning("No team column found in traditional boxscores dataset.")
        else:
            # Compute team season performance from boxscores
            wl_col = None
            for cand in ["wl", "result"]:
                if cand in df_recent.columns:
                    wl_col = cand
                    break

            if wl_col is not None:
                team_season = (
                    df_recent
                    .groupby([team_col, "season_year"])
                    .agg(
                        games=("pts", "size"),
                        avg_pts=("pts", "mean"),
                        avg_reb=("reb", "mean") if "reb" in df_recent.columns else ("pts", "mean"),
                        avg_ast=("ast", "mean") if "ast" in df_recent.columns else ("pts", "mean"),
                        wins=(wl_col, lambda x: np.sum(x.astype(str).str.startswith("W")))
                    )
                    .reset_index()
                )
                team_season["losses"] = team_season["games"] - team_season["wins"]
                team_season["win_pct_box"] = team_season["wins"] / team_season["games"]

                st.subheader(f"Team season summary (last {RECENT_YEARS_WINDOW} years, from boxscores)")
                st.dataframe(team_season.head(50))

            # Combine with season_records_df for official W/L
            if "team" in season_records_df.columns:
                # For join, we need a consistent team label.
                # This might not be perfect due to historical name changes.
                sr = season_records_df.copy()
                sr_recent = sr[sr["year"] >= recent_year_cut]
                st.subheader(f"Official season records (last {RECENT_YEARS_WINDOW} years)")
                st.dataframe(sr_recent.head(50))

            # Allow user to pick a team and see its trajectory
            team_choices = sorted(df_recent[team_col].dropna().unique())
            selected_team = st.selectbox("Select team", team_choices)
            ts_team = team_season[team_season[team_col] == selected_team] if "team_season" in locals() else pd.DataFrame()
            if not ts_team.empty:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(ts_team["season_year"], ts_team["win_pct_box"], marker="o")
                ax.set_ylim(0, 1)
                ax.set_title(f"{selected_team}: win% over last {RECENT_YEARS_WINDOW} seasons (from boxscores)")
                ax.set_xlabel("Season end year")
                ax.set_ylabel("Win%")
                st.pyplot(fig)


# -------------------------
# TAB 4: HALL OF FAME MODEL
# -------------------------
with tabs[4]:
    st.header("🏆 Hall of Fame Probability Model")

    st.markdown("""
This is a **heuristic statistical model** that combines:

- **Career value metrics**: total Win Shares, total VORP, total points  
- **Per-game impact**: points / rebounds / assists per game, PER  
- **Team success**: average team win%, number of title seasons, number of playoff seasons  

We:
1. Compute **z-scores** for each metric across all modern-era players  
2. Build a **weighted HoF score**  
3. Pass it through a **logistic transform** to get a value in [0,1], interpreted as a *probability-like Hall of Fame chance*  
4. Bucket players into tiers: **Unlikely**, **On the bubble**, **Strong case**, **Virtual lock**  
""")

    df_hof = career_df.copy()

    # Filters
    min_seasons = st.slider("Minimum seasons played", 1, 15, 3, 1)
    min_games = st.slider("Minimum games played", 10, 1500, 200, 10)

    subset = df_hof[(df_hof["seasons_played"] >= min_seasons) &
                    (df_hof["games_played"] >= min_games)].copy()

    tier_options = ["All"] + subset["hof_tier"].dropna().unique().tolist()
    selected_tier = st.selectbox("Filter by HoF tier", tier_options)

    if selected_tier != "All":
        subset = subset[subset["hof_tier"] == selected_tier]

    st.subheader("Top players by HoF probability")
    st.dataframe(
        subset[[
            "player_name", "hof_prob", "hof_tier",
            "seasons_played", "games_played",
            "pts_per_g", "trb_per_g", "ast_per_g",
            "total_ws", "total_vorp",
            "avg_per", "avg_team_win_pct",
            "title_seasons", "playoff_seasons",
            "first_year", "last_year"
        ]].sort_values("hof_prob", ascending=False).head(50)
    )

    # Scatter: career volume vs HoF probability
    st.subheader("Volume vs HoF probability")

    x_metric = st.selectbox(
        "X-axis metric",
        ["total_ws", "total_vorp", "total_pts", "games_played", "seasons_played"]
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(subset[x_metric], subset["hof_prob"], alpha=0.6)
    ax.set_xlabel(x_metric)
    ax.set_ylabel("HoF probability")
    ax.set_title(f"{x_metric} vs Hall of Fame probability")
    st.pyplot(fig)

    # Single-player HoF explanation
    st.markdown("### Inspect a single player's HoF stats")
    player_choices = sorted(df_hof["player_name"].dropna().unique())
    selected_player = st.selectbox("Select player", player_choices)
    row = df_hof[df_hof["player_name"] == selected_player].head(1)

    if not row.empty:
        r = row.iloc[0]
        st.write(f"**{selected_player}** — HoF probability: **{r.hof_prob:.3f}** ({r.hof_tier})")
        st.write(f"Seasons played: {int(r.seasons_played)} | Games played: {int(r.games_played)}")
        st.write(f"Career per-game: {r.pts_per_g:.1f} PTS, {r.trb_per_g:.1f} REB, {r.ast_per_g:.1f} AST")
        st.write(f"Total WS: {r.total_ws:.1f} | Total VORP: {r.total_vorp:.1f}")
        st.write(f"Average PER: {r.avg_per:.1f} | Average team win%: {r.avg_team_win_pct:.3f}")
        st.write(f"Title seasons: {int(r.title_seasons)} | Playoff seasons: {int(r.playoff_seasons)}")
        st.write(f"Modern era span: {int(r.first_year)}–{int(r.last_year)}")


# -------------------------
# TAB 5: MISSING DATA & IMPUTATION
# -------------------------
with tabs[5]:
    st.header("🔧 Missing Data & Imputation (Season-Level)")

    df = players_season_df.copy()
    miss = df.isna().mean().sort_values(ascending=False)
    st.subheader("Missingness fraction per column")
    st.dataframe(miss.to_frame("missing_fraction"))

    st.subheader("Interactive imputation demo")
    cols_to_impute = st.multiselect(
        "Select columns to impute",
        options=df.columns.tolist(),
        default=[c for c in df.select_dtypes(include=np.number).columns if df[c].isna().any()]
    )
    strategy = st.selectbox("Imputation strategy", ["mean", "median", "mode", "ffill", "bfill"])
    preview_rows = st.slider("Preview rows", 5, 50, 10, 5)

    df_imputed = df.copy()
    if cols_to_impute:
        for c in cols_to_impute:
            if strategy == "mean" and pd.api.types.is_numeric_dtype(df_imputed[c]):
                df_imputed[c] = df_imputed[c].fillna(df_imputed[c].mean())
            elif strategy == "median" and pd.api.types.is_numeric_dtype(df_imputed[c]):
                df_imputed[c] = df_imputed[c].fillna(df_imputed[c].median())
            elif strategy == "mode":
                if df_imputed[c].mode().empty:
                    df_imputed[c] = df_imputed[c].fillna(method="ffill").fillna(method="bfill")
                else:
                    df_imputed[c] = df_imputed[c].fillna(df_imputed[c].mode()[0])
            elif strategy == "ffill":
                df_imputed[c] = df_imputed[c].fillna(method="ffill")
            elif strategy == "bfill":
                df_imputed[c] = df_imputed[c].fillna(method="bfill")

    c1, c2 = st.columns(2)
    with c1:
        st.write("Before imputation")
        st.dataframe(df.head(preview_rows))
    with c2:
        st.write("After imputation")
        st.dataframe(df_imputed.head(preview_rows))

    # Simple validation: mean shift for numeric columns
    num_cols = [c for c in cols_to_impute if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        delta = (df_imputed[num_cols].mean() - df[num_cols].mean()).to_frame("mean_shift")
        st.subheader("Imputation sanity check (mean shift on numeric columns)")
        st.dataframe(delta)


# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown(
    "Data sources: Kaggle datasets "
    "`drgilermo/nba-players-stats`, "
    "`szymonjwiak/nba-traditional`, "
    "`boonpalipatana/nba-season-records-from-every-year`."
)
st.markdown("Created by **Aditya Sudarsan Anand** — CMSE 830 Final Project")
