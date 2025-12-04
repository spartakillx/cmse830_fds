# ==================================================
# CMSE 830 FINAL PROJECT
# NBA MULTI-DATASET ANALYTICS + HOF INDEX DASHBOARD
# ==================================================
import streamlit as st
import pandas as pd
import numpy as np
import os
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="NBA Analytics & Hall of Fame Index",
    page_icon="🏀",
    layout="wide"
)

st.title("🏀 NBA Analytics & Hall of Fame Index Dashboard")

st.markdown(
    """
This app combines **multiple NBA datasets** (boxscores, team season records, and player stats)
to build:

- Season & team-level exploration  
- Player comparison and trends  
- A **Hall of Fame Index (0–100)** that scores every player's career based on
  longevity, production, and team success.  

_Data sources: Kaggle – `drgilermo/nba-players-stats`, `szymonjwiak/nba-traditional`,
`boonpalipatana/nba-season-records-from-every-year`._
"""
)

# --------------------------------------------------
# HELPER: generic Kaggle loader
# --------------------------------------------------
def load_kaggle_csv(dataset_id: str) -> pd.DataFrame:
    """Download a Kaggle dataset with kagglehub and return the first CSV found."""
    path = kagglehub.dataset_download(dataset_id)
    files = [f for f in os.listdir(path) if f.lower().endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"No CSV files found in Kaggle dataset {dataset_id}")
    csv_path = os.path.join(path, files[0])
    return pd.read_csv(csv_path)


@st.cache_data(show_spinner=True)
def load_all_raw():
    """
    Load all three raw datasets from Kaggle.
    We do not assume exact filenames – only that each dataset has a single CSV.
    """
    players_raw = load_kaggle_csv("drgilermo/nba-players-stats")
    boxscores_raw = load_kaggle_csv("szymonjwiak/nba-traditional")
    seasons_raw = load_kaggle_csv("boonpalipatana/nba-season-records-from-every-year")

    return players_raw, boxscores_raw, seasons_raw


# --------------------------------------------------
# DATA WRANGLING: construct season & career tables
# --------------------------------------------------
@st.cache_data(show_spinner=True)
def build_clean_tables():
    players_raw, boxscores_raw, seasons_raw = load_all_raw()

    # --- 1) NORMALIZE COLUMN NAMES ---
    players = players_raw.copy()
    players.columns = players.columns.str.strip().str.lower()

    box = boxscores_raw.copy()
    box.columns = box.columns.str.strip().str.lower()

    seasons = seasons_raw.copy()
    seasons.columns = seasons.columns.str.strip().str.lower()

    # --------------------------------------------------
    # 1.a CAREER-LEVEL TABLE (from players_raw)
    # --------------------------------------------------
    career = players.copy()

    # Try to standardize player name column to "player_name"
    if "player" in career.columns and "player_name" not in career.columns:
        career = career.rename(columns={"player": "player_name"})

    # Some datasets use "from" / "to"
    if "from" in career.columns and "from_year" not in career.columns:
        career = career.rename(columns={"from": "from_year"})
    if "to" in career.columns and "to_year" not in career.columns:
        career = career.rename(columns={"to": "to_year"})

    # If "seasons" / "games" aren't present, try to infer:
    if "seasons" not in career.columns and "yrs" in career.columns:
        career = career.rename(columns={"yrs": "seasons"})
    if "games" not in career.columns and "g" in career.columns:
        career = career.rename(columns={"g": "games"})

    # Ensure numeric types for likely numeric columns
    numeric_cols_guess = [
        "from_year", "to_year", "seasons", "games",
        "tot_g", "tot_mp", "tot_pts", "tot_trb", "tot_ast",
        "tot_stl", "tot_blk", "tot_tov",
        "ws", "ws/48", "bpm", "vorp"
    ]
    for col in numeric_cols_guess:
        if col in career.columns:
            career[col] = pd.to_numeric(career[col], errors="coerce")

    if "ws/48" in career.columns and "ws_per_48" not in career.columns:
        career = career.rename(columns={"ws/48": "ws_per_48"})
    if "bpm" in career.columns and "avg_bpm" not in career.columns:
        career = career.rename(columns={"bpm": "avg_bpm"})

    # --------------------------------------------------
    # 1.b SEASON-LEVEL TABLE FROM BOXSCORES (ROBUST)
    # --------------------------------------------------
    rename_box = {}

    # season / year-like
    if "season" in box.columns:
        rename_box["season"] = "season"
    elif "season_id" in box.columns:
        rename_box["season_id"] = "season"
    elif "year" in box.columns:
        rename_box["year"] = "season"

    # player name
    if "player_name" in box.columns:
        rename_box["player_name"] = "player_name"
    elif "player" in box.columns:
        rename_box["player"] = "player_name"
    elif "name" in box.columns:
        rename_box["name"] = "player_name"

    # team
    if "team_abbreviation" in box.columns:
        rename_box["team_abbreviation"] = "team"
    elif "team" in box.columns:
        rename_box["team"] = "team"

    # stats
    if "pts" in box.columns:
        rename_box["pts"] = "pts"
    if "reb" in box.columns:
        rename_box["reb"] = "reb"
    if "ast" in box.columns:
        rename_box["ast"] = "ast"
    if "stl" in box.columns:
        rename_box["stl"] = "stl"
    if "blk" in box.columns:
        rename_box["blk"] = "blk"
    if "tov" in box.columns:
        rename_box["tov"] = "tov"
    if "min" in box.columns:
        rename_box["min"] = "min"
    if "mp" in box.columns and "min" not in rename_box:
        rename_box["mp"] = "min"

    # game id
    if "game_id" in box.columns:
        rename_box["game_id"] = "game_id"

    box = box.rename(columns=rename_box)

    keep_cols = [c for c in [
        "season", "player_name", "team", "game_id",
        "pts", "reb", "ast", "stl", "blk", "tov", "min"
    ] if c in box.columns]

    subset_cols = [c for c in ["season", "player_name"] if c in keep_cols]

    box = box[keep_cols]
    if subset_cols:
        box = box.dropna(subset=subset_cols)

    # Filter to 2005+ era
    if "season" in box.columns:
        if box["season"].dtype == object:
            def season_to_start_year(x):
                try:
                    s = str(x)
                    if "-" in s:
                        return int(s.split("-")[0])
                    return int(s)
                except Exception:
                    return np.nan

            box["season_start"] = box["season"].apply(season_to_start_year)
        else:
            box["season_start"] = box["season"]
    else:
        box["season_start"] = np.nan

    box = box.dropna(subset=["season_start"])
    box["season_start"] = box["season_start"].astype(int)
    box = box[box["season_start"] >= 2005]

    # Aggregate to player-season level
    agg_dict = {}
    for c in ["pts", "reb", "ast", "stl", "blk", "tov", "min"]:
        if c in box.columns:
            agg_dict[c] = "sum"
    if "game_id" in box.columns:
        agg_dict["game_id"] = pd.Series.nunique

    season_player = (
        box.groupby(["season_start", "player_name", "team"], as_index=False)
        .agg(agg_dict)
        .rename(columns={"season_start": "year", "game_id": "games"})
    )

    # --------------------------------------------------
    # 1.c TEAM SEASON RECORDS (wins / losses / win%)
    # --------------------------------------------------
    if "season" in seasons.columns and "year" not in seasons.columns:
        seasons = seasons.rename(columns={"season": "year"})
    if "team_name" in seasons.columns and "team" not in seasons.columns:
        seasons = seasons.rename(columns={"team_name": "team"})
    if "win%" in seasons.columns and "win_pct" not in seasons.columns:
        seasons = seasons.rename(columns={"win%": "win_pct"})

    for c in ["year", "wins", "losses", "win_pct"]:
        if c in seasons.columns:
            seasons[c] = pd.to_numeric(seasons[c], errors="coerce")

    if "year" in seasons.columns:
        seasons = seasons[seasons["year"] >= 2005]

    team_cols = ["year", "team"]
    if "win_pct" in seasons.columns:
        team_cols.append("win_pct")

    team_seasons = seasons[team_cols].drop_duplicates()

    # --------------------------------------------------
    # 1.d MERGE PLAYER-SEASON WITH TEAM WIN%
    # --------------------------------------------------
    season_merged = season_player.merge(
        team_seasons,
        on=["year", "team"],
        how="left"
    )

    # --------------------------------------------------
    # 1.e BUILD CAREER TABLE FROM SEASON-MERGED
    # --------------------------------------------------
    career_agg_dict = {
        "year": ["min", "max", "nunique"]
    }
    
    if "games" in season_merged.columns:
        career_agg_dict["games"] = "sum"
    
    for stat_col in ["pts", "reb", "ast", "stl", "blk", "tov"]:
        if stat_col in season_merged.columns:
            career_agg_dict[stat_col] = "sum"
    
    if "win_pct" in season_merged.columns:
        career_agg_dict["win_pct"] = "mean"
    
    career_from_box = season_merged.groupby("player_name").agg(career_agg_dict).reset_index()
    
    if isinstance(career_from_box.columns, pd.MultiIndex):
        career_from_box.columns = ['_'.join(str(c) for c in col).strip('_') if col[1] else col[0] 
                                    for col in career_from_box.columns.values]
    
    rename_dict = {
        "year_min": "from_year",
        "year_max": "to_year",
        "year_nunique": "seasons"
    }
    
    if "games_sum" in career_from_box.columns:
        rename_dict["games_sum"] = "games"
    
    for stat in ["pts", "reb", "ast", "stl", "blk", "tov"]:
        if f"{stat}_sum" in career_from_box.columns:
            rename_dict[f"{stat}_sum"] = f"tot_{stat}"
    
    if "win_pct_mean" in career_from_box.columns:
        rename_dict["win_pct_mean"] = "avg_team_win_pct"
    
    career_from_box = career_from_box.rename(columns=rename_dict)
    
    if "games" not in career_from_box.columns:
        career_from_box["games"] = 0

    if "player_name" in career.columns:
        base_cols = ["player_name"]
        extra_cols = [c for c in career.columns if c not in base_cols]
        career_extra = career[base_cols + extra_cols].drop_duplicates(subset=["player_name"])
        career_all = career_from_box.merge(
            career_extra,
            on="player_name",
            how="left",
            suffixes=("", "_career")
        )
    else:
        career_all = career_from_box.copy()

    for c in career_all.columns:
        if career_all[c].dtype == "object":
            maybe_num = pd.to_numeric(career_all[c], errors="ignore")
            career_all[c] = maybe_num

    ordering = [c for c in [
        "player_name", "from_year", "to_year", "seasons", "games",
        "tot_pts", "tot_reb", "tot_ast", "tot_stl", "tot_blk", "tot_tov",
        "avg_team_win_pct"
    ] if c in career_all.columns] + [
        c for c in career_all.columns if c not in [
            "player_name", "from_year", "to_year", "seasons", "games",
            "tot_pts", "tot_reb", "tot_ast", "tot_stl", "tot_blk", "tot_tov",
            "avg_team_win_pct"
        ]
    ]
    career_all = career_all[ordering]

    return season_merged, team_seasons, career_all


season_df, team_df, career_df = build_clean_tables()

# --------------------------------------------------
# HOF INDEX (0–100) FOR ALL PLAYERS
# --------------------------------------------------
def add_hof_index(career: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a Hall-of-Fame style index using *all* players.
    """
    df = career.copy()

    feature_cols = [c for c in [
        "seasons", "games", "tot_pts", "tot_reb", "tot_ast", "avg_team_win_pct"
    ] if c in df.columns]

    if not feature_cols:
        df["hof_index"] = 0.0
        return df

    X = df[feature_cols].fillna(0).astype(float)

    # z-score
    z = (X - X.mean()) / (X.std(ddof=0) + 1e-10)
    z = z.fillna(0.0)

    raw_score = z.sum(axis=1)
    
    # Convert to 0-100 percentile
    ranks = raw_score.rank(method="average", pct=True)
    df["hof_index"] = (ranks * 100).astype(float)

    return df


career_df = add_hof_index(career_df)

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

# ==================================================
# TAB 1: OVERVIEW
# ==================================================
with tabs[0]:
    st.subheader("Dataset overview")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Season-level player stats (merged with team win% where available):**")
        st.write(season_df.head())
        st.write(f"Rows: {len(season_df)} | Columns: {season_df.shape[1]}")
        st.markdown("**Columns (first 40):**")
        st.code(", ".join(season_df.columns.tolist()[:40]))

    with c2:
        st.markdown("**Career-level table (aggregated, with HoF index):**")
        st.write(career_df.head())
        st.write(f"Players: {len(career_df)}")
        st.markdown("**Columns (first 40):**")
        st.code(", ".join(career_df.columns.tolist()[:40]))

    st.markdown("---")
    st.markdown(
        "Created by **Aditya Sudarsan Anand** – CMSE 830 Final Project.  "
        "Data sources: `drgilermo/nba-players-stats`, `szymonjwiak/nba-traditional`, "
        "`boonpalipatana/nba-season-records-from-every-year`."
    )

# ==================================================
# TAB 2: EDA (Season & Team)
# ==================================================
with tabs[1]:
    st.subheader("Season & Team EDA")

    numeric_cols = season_df.select_dtypes(include=np.number).columns.tolist()

    colA, colB = st.columns(2)

    with colA:
        st.markdown("**Correlation heatmap (season-level numeric stats)**")
        if len(numeric_cols) >= 2:
            method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"], index=0)
            corr = season_df[numeric_cols].corr(method=method)
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
            ax.set_title(f"Correlation ({method})")
            st.pyplot(fig)
        else:
            st.info("Not enough numeric columns for correlation heatmap.")

    with colB:
        st.markdown("**Distribution of a season-level stat**")
        if numeric_cols:
            stat = st.selectbox("Select numeric stat", numeric_cols, index=0)
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.histplot(season_df[stat].dropna(), kde=True, bins=30, ax=ax)
            ax.set_title(f"Distribution of {stat}")
            st.pyplot(fig)
        else:
            st.info("No numeric columns detected.")

    st.markdown("---")
    st.markdown("### Team win% over seasons")

    if "year" in team_df.columns and "win_pct" in team_df.columns:
        team_list = sorted(team_df["team"].dropna().unique())
        sel_team = st.selectbox("Choose a team", team_list)
        tdf = team_df[team_df["team"] == sel_team].sort_values("year")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(tdf["year"], tdf["win_pct"], marker="o")
        ax.set_xlabel("Year")
        ax.set_ylabel("Win%")
        ax.set_title(f"{sel_team} – win% over time")
        st.pyplot(fig)
    else:
        st.info("Team win% columns not found in team dataset.")

# ==================================================
# TAB 3: PLAYER EXPLORER
# ==================================================
with tabs[2]:
    st.subheader("Player explorer (season-level)")

    all_players_season = sorted(season_df["player_name"].dropna().unique())
    sel_player = st.selectbox("Select a player", all_players_season)

    pdf = season_df[season_df["player_name"] == sel_player].sort_values("year")

    st.markdown(f"### {sel_player} – season overview")
    st.dataframe(pdf)

    stat_choices = [c for c in ["pts", "reb", "ast"] if c in pdf.columns]
    if stat_choices:
        stat_to_plot = st.selectbox("Plot stat over time", stat_choices, index=0)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(pdf["year"], pdf[stat_to_plot], marker="o")
        ax.set_xlabel("Year")
        ax.set_ylabel(stat_to_plot)
        ax.set_title(f"{sel_player} – {stat_to_plot} over seasons")
        st.pyplot(fig)

# ==================================================
# TAB 4: PLAYER COMPARISON
# ==================================================
with tabs[3]:
    st.subheader("Player Comparison")
    
    st.markdown("Compare two players' career stats and seasonal performance")
    
    all_players_comp = sorted(career_df["player_name"].dropna().unique())
    
    col1, col2 = st.columns(2)
    
    with col1:
        player1 = st.selectbox("Select Player 1", all_players_comp, key="p1")
    with col2:
        player2 = st.selectbox("Select Player 2", all_players_comp, 
                               index=min(1, len(all_players_comp)-1), key="p2")
    
    # Career comparison
    st.markdown("### Career Statistics Comparison")
    
    p1_career = career_df[career_df["player_name"] == player1].iloc[0] if not career_df[career_df["player_name"] == player1].empty else None
    p2_career = career_df[career_df["player_name"] == player2].iloc[0] if not career_df[career_df["player_name"] == player2].empty else None
    
    if p1_career is not None and p2_career is not None:
        comp_stats = ["seasons", "games", "tot_pts", "tot_reb", "tot_ast", "hof_index"]
        comp_data = []
        
        for stat in comp_stats:
            if stat in p1_career.index and stat in p2_career.index:
                val1 = p1_career[stat]
                val2 = p2_career[stat]
                
                val1 = val1 if not pd.isna(val1) else 0
                val2 = val2 if not pd.isna(val2) else 0
                
                comp_data.append({
                    "Statistic": stat.replace("tot_", "Total ").replace("_", " ").title(),
                    player1: f"{val1:.1f}" if stat == "hof_index" else int(val1),
                    player2: f"{val2:.1f}" if stat == "hof_index" else int(val2)
                })
        
        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, use_container_width=True)
        
        # Visual comparison
        st.markdown("### Per-Game Averages Comparison")
        
        if "tot_pts" in p1_career.index and "games" in p1_career.index:
            viz_stats = []
            
            for stat_name, tot_col in [("PPG", "tot_pts"), ("RPG", "tot_reb"), ("APG", "tot_ast")]:
                if tot_col in p1_career.index and tot_col in p2_career.index:
                    p1_games = p1_career["games"] if not pd.isna(p1_career["games"]) and p1_career["games"] > 0 else 1
                    p2_games = p2_career["games"] if not pd.isna(p2_career["games"]) and p2_career["games"] > 0 else 1
                    
                    p1_avg = (p1_career[tot_col] if not pd.isna(p1_career[tot_col]) else 0) / p1_games
                    p2_avg = (p2_career[tot_col] if not pd.isna(p2_career[tot_col]) else 0) / p2_games
                    
                    viz_stats.append({
                        "Stat": stat_name,
                        player1: p1_avg,
                        player2: p2_avg
                    })
            
            if viz_stats:
                viz_df = pd.DataFrame(viz_stats)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                x = np.arange(len(viz_df))
                width = 0.35
                
                ax.bar(x - width/2, viz_df[player1], width, label=player1, alpha=0.8)
                ax.bar(x + width/2, viz_df[player2], width, label=player2, alpha=0.8)
                
                ax.set_xlabel('Statistic')
                ax.set_ylabel('Per Game Average')
                ax.set_title('Career Per-Game Averages')
                ax.set_xticks(x)
                ax.set_xticklabels(viz_df["Stat"])
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                
                st.pyplot(fig)
    
    # Season-by-season comparison
    st.markdown("### Season-by-Season Comparison")
    
    p1_seasons = season_df[season_df["player_name"] == player1].sort_values("year")
    p2_seasons = season_df[season_df["player_name"] == player2].sort_values("year")
    
    if not p1_seasons.empty and not p2_seasons.empty:
        stat_to_compare = st.selectbox(
            "Select stat to compare over time",
            [c for c in ["pts", "reb", "ast", "stl", "blk"] if c in season_df.columns]
        )
        
        if stat_to_compare:
            fig, ax = plt.subplots(figsize=(12, 5))
            
            ax.plot(p1_seasons["year"], p1_seasons[stat_to_compare], 
                   marker="o", label=player1, linewidth=2)
            ax.plot(p2_seasons["year"], p2_seasons[stat_to_compare], 
                   marker="s", label=player2, linewidth=2)
            
            ax.set_xlabel("Season")
            ax.set_ylabel(stat_to_compare.upper())
            ax.set_title(f"{stat_to_compare.upper()} Comparison Over Time")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)

# ==================================================
# TAB 5: TEAM TRENDS
# ==================================================
with tabs[4]:
    st.subheader("Team trends (from season-level stats)")

    if "team" in season_df.columns:
        team_list2 = sorted(season_df["team"].dropna().unique())
        sel_team2 = st.selectbox("Select team", team_list2)
        tdf2 = season_df[season_df["team"] == sel_team2]

        agg_cols2 = {}
        for c in ["pts", "reb", "ast"]:
            if c in tdf2.columns:
                agg_cols2[c] = "sum"
        
        if "games" in tdf2.columns:
            agg_cols2["games"] = "sum"
        
        if agg_cols2:
            team_season_stats = (
                tdf2.groupby("year")
                .agg(agg_cols2)
                .reset_index()
            )
        else:
            team_season_stats = (
                tdf2.groupby("year")
                .size()
                .reset_index(name="count")
            )

        st.markdown(f"### {sel_team2} – season totals (aggregated from boxscores)")
        st.dataframe(team_season_stats)

        if "pts" in team_season_stats.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(team_season_stats["year"], team_season_stats["pts"], marker="o")
            ax.set_xlabel("Year")
            ax.set_ylabel("Total points (sum of boxscores)")
            ax.set_title(f"{sel_team2} – total points per season")
            st.pyplot(fig)
    else:
        st.info("No 'team' column in season-level data.")

# ==================================================
# TAB 6: HALL OF FAME EXPLORER
# ==================================================
with tabs[5]:
    st.subheader("Hall of Fame Index Explorer (0-100 scale)")

    st.markdown(
        """
**Hall of Fame Index** is a relative, data-driven score (0-100) built from:

- Career **seasons** and **games** (longevity)  
- Total **points**, **rebounds**, **assists** (production)  
- Average **team win%** over the player's seasons (team success)

We standardize each metric, sum them, and convert to a **percentile (0-100)**  
across *all players in the dataset*. Higher = more Hall-of-Fame-like profile.
"""
    )

    if career_df.empty:
        st.warning("Career table is empty – check data loading.")
    else:
        st.write(f"**Total players in dataset: {len(career_df):,}**")
        st.write(f"**Available columns:** {', '.join(career_df.columns.tolist()[:15])}")
        
        # Filter options
        col1, col2 = st.columns(2)
        
        has_seasons = "seasons" in career_df.columns
        has_games = "games" in career_df.columns
        
        with col1:
            if has_seasons:
                max_s = int(career_df["seasons"].max()) if career_df["seasons"].notna().any() else 20
                min_seasons = st.slider("Minimum seasons", 0, max_s, 0)
            else:
                min_seasons = 0
                st.info("'seasons' column not found")
        
        with col2:
            if has_games:
                max_g = int(career_df["games"].max()) if career_df["games"].notna().any() else 1500
                min_games = st.slider("Minimum games", 0, max_g, 0, step=50)
            else:
                min_games = 0
                st.info("'games' column not found")
        
        # Apply filters
        career_filtered = career_df.copy()
        if has_seasons and min_seasons > 0:
            career_filtered = career_filtered[career_filtered["seasons"].fillna(0) >= min_seasons]
        if has_games and min_games > 0:
            career_filtered = career_filtered[career_filtered["games"].fillna(0) >= min_games]
        
        if "hof_index" in career_filtered.columns:
            career_filtered = career_filtered.sort_values("hof_index", ascending=False)
        
        st.write
        if len(career_filtered) == 0:
        st.warning("No players match filters. Lower the thresholds.")
    else:
        # Top N
        max_slider = max(10, len(career_filtered))
        top_n = st.slider("Show top N", 10, min(200, max_slider), min(50, max_slider), 10)

        st.markdown(f"### Top {top_n} players by HoF Index")
        
        display_cols = [c for c in [
            "player_name", "from_year", "to_year", "seasons", "games",
            "tot_pts", "tot_reb", "tot_ast", "avg_team_win_pct", "hof_index"
        ] if c in career_filtered.columns]
        
        if display_cols:
            top_players = career_filtered[display_cols].head(top_n).copy()
            if "hof_index" in top_players.columns:
                top_players["hof_index"] = top_players["hof_index"].round(1)
            st.dataframe(top_players, use_container_width=True)
        else:
            st.error("No display columns available")

    # Player inspector
    st.markdown("---")
    st.markdown("### Inspect individual player")
    
    if "player_name" in career_df.columns:
        all_players = sorted(career_df["player_name"].dropna().unique())
        
        if all_players:
            sel = st.selectbox(f"Select from ALL {len(all_players):,} players", all_players, key="hof_pl")
            
            row = career_df[career_df["player_name"] == sel]
            if not row.empty:
                r = row.iloc[0]
                
                st.markdown(f"## {sel}")
                
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.markdown("**Career Overview**")
                    if "from_year" in r.index and pd.notna(r["from_year"]):
                        st.write(f"From: {int(r['from_year'])}")
                    if "to_year" in r.index and pd.notna(r["to_year"]):
                        st.write(f"To: {int(r['to_year'])}")
                    if "seasons" in r.index and pd.notna(r["seasons"]):
                        st.write(f"Seasons: {int(r['seasons'])}")
                    if "games" in r.index and pd.notna(r["games"]):
                        st.write(f"Games: {int(r['games']):,}")
                
                with c2:
                    st.markdown("**Career Totals**")
                    if "tot_pts" in r.index and pd.notna(r["tot_pts"]):
                        st.write(f"Points: {int(r['tot_pts']):,}")
                    if "tot_reb" in r.index and pd.notna(r["tot_reb"]):
                        st.write(f"Rebounds: {int(r['tot_reb']):,}")
                    if "tot_ast" in r.index and pd.notna(r["tot_ast"]):
                        st.write(f"Assists: {int(r['tot_ast']):,}")
                
                with c3:
                    st.markdown("**Team Success**")
                    if "avg_team_win_pct" in r.index and pd.notna(r["avg_team_win_pct"]):
                        st.write(f"Avg team win%: {r['avg_team_win_pct']:.3f}")
                
                # HoF Index
                hof_idx = float(r.get("hof_index", 0.0))
                st.markdown(f"### Hall of Fame Index: **{hof_idx:.1f} / 100**")
                
                if hof_idx >= 95:
                    verdict = "🏆 Elite / Inner-circle HoF"
                    color = "green"
                elif hof_idx >= 85:
                    verdict = "⭐ Strong HoF candidate"
                    color = "blue"
                elif hof_idx >= 70:
                    verdict = "🎯 Borderline HoF"
                    color = "orange"
                elif hof_idx >= 50:
                    verdict = "✅ Solid career"
                    color = "gray"
                else:
                    verdict = "📊 Role player"
                    color = "lightgray"
                
                st.markdown(f"**{verdict}**")
                
                fig, ax = plt.subplots(figsize=(8, 1.5))
                ax.barh([0], [hof_idx], height=0.5, color=color)
                ax.set_xlim(0, 100)
                ax.set_ylim(-0.5, 0.5)
                ax.set_yticks([])
                ax.set_xlabel("HoF Index (0-100)")
                ax.set_title(f"{sel} - HoF Index")
                ax.axvline(x=50, color='gray', linestyle='--', alpha=0.3)
                ax.axvline(x=85, color='blue', linestyle='--', alpha=0.3)
                ax.axvline(x=95, color='green', linestyle='--', alpha=0.3)
                st.pyplot(fig)
    else:
        st.error("player_name column not found")

# Download
st.markdown("---")
st.markdown("### Download cleaned career-level table")

def to_csv_bytes(df_in: pd.DataFrame) -> bytes:
    return df_in.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Download career table with HoF Index",
    data=to_csv_bytes(career_df),
    file_name="career_with_hof_index.csv",
    mime="text/csv"
)</parameter>
