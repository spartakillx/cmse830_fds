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
    """Load all three raw datasets from Kaggle."""
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

    # --- normalize column names ---
    players = players_raw.copy()
    players.columns = players.columns.str.strip().str.lower()

    box = boxscores_raw.copy()
    box.columns = box.columns.str.strip().str.lower()

    seasons = seasons_raw.copy()
    seasons.columns = seasons.columns.str.strip().str.lower()

    # --------------------------------------------------
    # 1) CAREER-LEVEL TABLE (from players_raw)
    # --------------------------------------------------
    career = players.copy()

    if "player" in career.columns and "player_name" not in career.columns:
        career = career.rename(columns={"player": "player_name"})

    if "from" in career.columns and "from_year" not in career.columns:
        career = career.rename(columns={"from": "from_year"})
    if "to" in career.columns and "to_year" not in career.columns:
        career = career.rename(columns={"to": "to_year"})

    if "seasons" not in career.columns and "yrs" in career.columns:
        career = career.rename(columns={"yrs": "seasons"})
    if "games" not in career.columns and "g" in career.columns:
        career = career.rename(columns={"g": "games"})

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
    # 2) SEASON-LEVEL TABLE FROM BOXSCORES
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

    # Map season to starting year
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
    box = box[box["season_start"] >= 2005]  # post–hand-check era

    # Aggregate to player-season level
    agg_dict = {}
    for c in ["pts", "reb", "ast", "stl", "blk", "tov", "min"]:
        if c in box.columns:
            agg_dict[c] = "sum"
    if "game_id" in box.columns:
        agg_dict["game_id"] = pd.Series.nunique  # games played

    season_player = (
        box.groupby(["season_start", "player_name", "team"], as_index=False)
        .agg(agg_dict)
        .rename(columns={"season_start": "year", "game_id": "games"})
    )

    # --------------------------------------------------
    # 3) TEAM SEASON RECORDS (wins / losses / win%)
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

    team_cols = [c for c in ["year", "team", "win_pct"] if c in seasons.columns]
    team_seasons = seasons[team_cols].drop_duplicates()

    # --------------------------------------------------
    # 4) MERGE PLAYER-SEASON WITH TEAM WIN%
    # --------------------------------------------------
    season_merged = season_player.merge(
        team_seasons,
        on=[c for c in ["year", "team"] if c in season_player.columns and c in team_seasons.columns],
        how="left"
    )

    # --------------------------------------------------
    # 5) BUILD CAREER TABLE FROM SEASON-MERGED
    # --------------------------------------------------
    cols_sm = season_merged.columns

    named_aggs = {
        "from_year": ("year", "min"),
        "to_year": ("year", "max"),
        "seasons": ("year", "nunique"),
    }

    if "games" in cols_sm:
        named_aggs["games"] = ("games", "sum")
    else:
        named_aggs["games"] = ("year", "size")

    if "pts" in cols_sm:
        named_aggs["tot_pts"] = ("pts", "sum")
    if "reb" in cols_sm:
        named_aggs["tot_reb"] = ("reb", "sum")
    if "ast" in cols_sm:
        named_aggs["tot_ast"] = ("ast", "sum")
    if "stl" in cols_sm:
        named_aggs["tot_stl"] = ("stl", "sum")
    if "blk" in cols_sm:
        named_aggs["tot_blk"] = ("blk", "sum")
    if "tov" in cols_sm:
        named_aggs["tot_tov"] = ("tov", "sum")
    if "win_pct" in cols_sm:
        named_aggs["avg_team_win_pct"] = ("win_pct", "mean")

    career_from_box = (
        season_merged.groupby("player_name")
        .agg(**named_aggs)
        .reset_index()
    )

    # merge extra career info from players dataset if names align
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
    Construct a Hall-of-Fame style index using all players.
    Features:
      - seasons, games
      - tot_pts, tot_reb, tot_ast
      - avg_team_win_pct (if available)
    Then convert to percentile [0, 100].
    """
    df = career.copy()

    feature_cols = [c for c in [
        "seasons", "games", "tot_pts", "tot_reb", "tot_ast", "avg_team_win_pct"
    ] if c in df.columns]

    X = df[feature_cols].fillna(0).astype(float)

    z = (X - X.mean()) / X.std(ddof=0)
    z = z.fillna(0.0)

    raw_score = z.sum(axis=1)
    ranks = raw_score.rank(method="average", pct=True)
    df["hof_index"] = (ranks * 100).astype(float)
    df["hof_index_rounded"] = df["hof_index"].round(1)

    return df


career_df = add_hof_index(career_df)

# --------------------------------------------------
# TABS
# --------------------------------------------------
tabs = st.tabs([
    "Overview",
    "EDA (Season & Team)",
    "Player Explorer",
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

    if "year" in team_df.columns and "win_pct" in team_df.columns and "team" in team_df.columns:
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
        st.info("Team win% columns not found in team dataset – adjust column names in build_clean_tables() if needed.")

# ==================================================
# TAB 3: PLAYER EXPLORER
# ==================================================
with tabs[2]:
    st.subheader("Player explorer (season-level)")

    if "player_name" in season_df.columns:
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
    else:
        st.info("No `player_name` column in season_df.")

# ==================================================
# TAB 4: TEAM TRENDS
# ==================================================
with tabs[3]:
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
        else:
            agg_cols2["games"] = "size"

        team_season_stats = (
            tdf2.groupby("year")
            .agg(agg_cols2)
            .reset_index()
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
        st.info("No 'team' column in season-level data. Check column mappings in build_clean_tables().")

# ==================================================
# TAB 5: HALL OF FAME EXPLORER
# ==================================================
with tabs[4]:
    st.subheader("Hall of Fame Explorer (Index 0–100)")

    st.markdown(
        """
**Hall of Fame Index** is a relative, data-driven score built from:

- Career **seasons** and **games** (longevity)  
- Total **points**, **rebounds**, **assists** (production)  
- Average **team win%** over the player's seasons (if available)  

We standardize each metric, sum them, and then convert the result to a **percentile (0–100)**  
across *all players in the dataset*. Higher = more Hall-of-Fame-like profile.
"""
    )

    min_seasons = st.slider("Min seasons", min_value=1, max_value=20, value=3)
    min_games = st.slider("Min career games", min_value=1, max_value=1500, value=100, step=50)

    career_eligible = career_df.copy()
    if "seasons" in career_eligible.columns:
        career_eligible = career_eligible[career_eligible["seasons"] >= min_seasons]
    if "games" in career_eligible.columns:
        career_eligible = career_eligible[career_eligible["games"] >= min_games]

    if "hof_index" not in career_eligible.columns:
        career_eligible = add_hof_index(career_eligible)

    career_eligible = career_eligible.sort_values("hof_index", ascending=False)

    # Player inspector
    all_players_career = sorted(career_eligible["player_name"].dropna().unique())
    st.markdown("### Inspect a player")
    if all_players_career:
        sel_player_hof = st.selectbox("Choose player", all_players_career)

        prow = career_eligible[career_eligible["player_name"] == sel_player_hof]
        if not prow.empty:
            prow = prow.iloc[0]

            st.markdown(f"#### {sel_player_hof}")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write("Seasons:", int(prow.get("seasons", np.nan)) if not pd.isna(prow.get("seasons", np.nan)) else "NA")
                st.write("Games:", int(prow.get("games", np.nan)) if not pd.isna(prow.get("games", np.nan)) else "NA")
            with c2:
                st.write("Total points:", int(prow.get("tot_pts", np.nan)) if not pd.isna(prow.get("tot_pts", np.nan)) else "NA")
                st.write("Total rebounds:", int(prow.get("tot_reb", np.nan)) if not pd.isna(prow.get("tot_reb", np.nan)) else "NA")
                st.write("Total assists:", int(prow.get("tot_ast", np.nan)) if not pd.isna(prow.get("tot_ast", np.nan)) else "NA")
            with c3:
                hof_val = float(prow["hof_index"])
                st.write("HoF Index (0–100):", f"{hof_val:.1f}")
                if hof_val >= 90:
                    verdict = "🚀 Inner-circle HoF profile"
                elif hof_val >= 75:
                    verdict = "⭐ Strong HoF-like profile"
                elif hof_val >= 50:
                    verdict = "🙂 Solid career"
                else:
                    verdict = "📈 Role player / limited impact"
                st.write("Verdict:", verdict)

    # Top 100
    st.markdown("### Top players by HoF Index")
    cols_to_show = [c for c in [
        "player_name", "from_year", "to_year", "seasons", "games",
        "tot_pts", "tot_reb", "tot_ast", "avg_team_win_pct",
        "hof_index_rounded"
    ] if c in career_eligible.columns]

    st.dataframe(
        career_eligible[cols_to_show]
        .rename(columns={"hof_index_rounded": "hof_index"})
        .reset_index(drop=True)
        .head(100)
    )

    # Download cleaned career table
    st.markdown("### Download cleaned career-level table")

    def to_csv_bytes(df_to_save: pd.DataFrame) -> bytes:
        return df_to_save.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download career table with HoF Index",
        data=to_csv_bytes(career_df),
        file_name="career_with_hof_index.csv",
        mime="text/csv"
    )
