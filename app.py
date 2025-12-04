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

    # Normalize column names
    players = players_raw.copy()
    players.columns = players.columns.str.strip().str.lower()

    box = boxscores_raw.copy()
    box.columns = box.columns.str.strip().str.lower()

    seasons = seasons_raw.copy()
    seasons.columns = seasons.columns.str.strip().str.lower()

    # Career-level table from players dataset
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
        "tot_stl", "tot_blk", "tot_tov", "ws", "ws/48", "bpm", "vorp"
    ]
    for col in numeric_cols_guess:
        if col in career.columns:
            career[col] = pd.to_numeric(career[col], errors="coerce")

    # Season-level from boxscores
    rename_box = {}
    if "season" in box.columns:
        rename_box["season"] = "season"
    elif "season_id" in box.columns:
        rename_box["season_id"] = "season"
    elif "year" in box.columns:
        rename_box["year"] = "season"

    if "player_name" in box.columns:
        rename_box["player_name"] = "player_name"
    elif "player" in box.columns:
        rename_box["player"] = "player_name"

    if "team_abbreviation" in box.columns:
        rename_box["team_abbreviation"] = "team"
    elif "team" in box.columns:
        rename_box["team"] = "team"

    for stat in ["pts", "reb", "ast", "stl", "blk", "tov"]:
        if stat in box.columns:
            rename_box[stat] = stat
    
    if "min" in box.columns:
        rename_box["min"] = "min"
    elif "mp" in box.columns:
        rename_box["mp"] = "min"
    
    if "game_id" in box.columns:
        rename_box["game_id"] = "game_id"

    box = box.rename(columns=rename_box)

    keep_cols = [c for c in ["season", "player_name", "team", "game_id", 
                              "pts", "reb", "ast", "stl", "blk", "tov", "min"] 
                 if c in box.columns]
    box = box[keep_cols]
    
    subset_cols = [c for c in ["season", "player_name"] if c in box.columns]
    if subset_cols:
        box = box.dropna(subset=subset_cols)

    # Convert season to year
    if "season" in box.columns:
        def season_to_year(x):
            try:
                s = str(x)
                if "-" in s:
                    return int(s.split("-")[0])
                return int(s)
            except:
                return np.nan
        
        if box["season"].dtype == object:
            box["season_start"] = box["season"].apply(season_to_year)
        else:
            box["season_start"] = box["season"]
    else:
        box["season_start"] = np.nan

    box = box.dropna(subset=["season_start"])
    box["season_start"] = box["season_start"].astype(int)
    box = box[box["season_start"] >= 2005]

    # Aggregate to player-season
    agg_dict = {}
    for c in ["pts", "reb", "ast", "stl", "blk", "tov", "min"]:
        if c in box.columns:
            agg_dict[c] = "sum"
    if "game_id" in box.columns:
        agg_dict["game_id"] = pd.Series.nunique

    if agg_dict:
        season_player = (
            box.groupby(["season_start", "player_name", "team"], as_index=False)
            .agg(agg_dict)
            .rename(columns={"season_start": "year", "game_id": "games"})
        )
    else:
        season_player = pd.DataFrame(columns=["year", "player_name", "team"])

    # Team records
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

    # Merge team win%
    season_merged = season_player.merge(team_seasons, on=["year", "team"], how="left")

    # Build career from season data
    career_agg_dict = {}
    if "year" in season_merged.columns:
        career_agg_dict["from_year"] = ("year", "min")
        career_agg_dict["to_year"] = ("year", "max")
        career_agg_dict["seasons"] = ("year", "nunique")
    
    if "games" in season_merged.columns:
        career_agg_dict["games"] = ("games", "sum")
    
    for stat in ["pts", "reb", "ast", "stl", "blk", "tov"]:
        if stat in season_merged.columns:
            career_agg_dict[f"tot_{stat}"] = (stat, "sum")
    
    if "win_pct" in season_merged.columns:
        career_agg_dict["avg_team_win_pct"] = ("win_pct", "mean")

    if career_agg_dict and "player_name" in season_merged.columns:
        career_from_box = season_merged.groupby("player_name").agg(**career_agg_dict).reset_index()
    else:
        career_from_box = pd.DataFrame(columns=["player_name"])

    # Merge with original career data
    if "player_name" in career.columns and not career_from_box.empty:
        career_extra = career[["player_name"] + [c for c in career.columns if c != "player_name"]].drop_duplicates(subset=["player_name"])
        career_all = career_from_box.merge(career_extra, on="player_name", how="left", suffixes=("", "_orig"))
    else:
        career_all = career_from_box.copy()

    return season_merged, team_seasons, career_all


season_df, team_df, career_df = build_clean_tables()

# --------------------------------------------------
# HOF INDEX (0–100) FOR ALL PLAYERS
# --------------------------------------------------
def add_hof_index(career: pd.DataFrame) -> pd.DataFrame:
    """Construct a Hall-of-Fame index (0-100) for all players."""
    df = career.copy()

    feature_cols = [c for c in ["seasons", "games", "tot_pts", "tot_reb", "tot_ast", "avg_team_win_pct"] 
                    if c in df.columns]

    if not feature_cols:
        df["hof_index"] = 0.0
        return df

    X = df[feature_cols].fillna(0).astype(float)
    
    # Z-score normalization
    mean = X.mean()
    std = X.std(ddof=0) + 1e-10
    z = (X - mean) / std
    z = z.fillna(0.0)

    raw_score = z.sum(axis=1)
    
    # Convert to 0-100 percentile
    ranks = raw_score.rank(method="average", pct=True)
    df["hof_index"] = (ranks * 100).round(1)

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

# TAB 1: OVERVIEW
with tabs[0]:
    st.subheader("Dataset overview")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Season-level player stats:**")
        st.write(season_df.head())
        st.write(f"Rows: {len(season_df):,} | Columns: {season_df.shape[1]}")

    with c2:
        st.markdown("**Career-level table:**")
        st.write(career_df.head())
        st.write(f"Players: {len(career_df):,}")

    st.markdown("---")
    st.markdown("Created by **Aditya Sudarsan Anand** – CMSE 830 Final Project.")

# TAB 2: EDA
with tabs[1]:
    st.subheader("Season & Team EDA")
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
            st.pyplot(fig)

    with colB:
        st.markdown("**Distribution**")
        if numeric_cols:
            stat = st.selectbox("Select stat", numeric_cols, index=0)
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.histplot(season_df[stat].dropna(), kde=True, bins=30, ax=ax)
            ax.set_title(f"Distribution of {stat}")
            st.pyplot(fig)

    st.markdown("---")
    st.markdown("### Team win% over seasons")

    if "year" in team_df.columns and "win_pct" in team_df.columns and "team" in team_df.columns:
        team_list = sorted(team_df["team"].dropna().unique())
        if team_list:
            sel_team = st.selectbox("Choose a team", team_list)
            tdf = team_df[team_df["team"] == sel_team].sort_values("year")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(tdf["year"], tdf["win_pct"], marker="o")
            ax.set_xlabel("Year")
            ax.set_ylabel("Win%")
            ax.set_title(f"{sel_team} – win% over time")
            st.pyplot(fig)

# TAB 3: PLAYER EXPLORER
with tabs[2]:
    st.subheader("Player explorer (season-level)")

    if "player_name" in season_df.columns:
        all_players_season = sorted(season_df["player_name"].dropna().unique())
        if all_players_season:
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

# TAB 4: PLAYER COMPARISON
with tabs[3]:
    st.subheader("Player Comparison")
    
    if "player_name" in career_df.columns:
        all_players_comp = sorted(career_df["player_name"].dropna().unique())
        
        if len(all_players_comp) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                player1 = st.selectbox("Select Player 1", all_players_comp, key="p1")
            with col2:
                player2 = st.selectbox("Select Player 2", all_players_comp, index=1, key="p2")
            
            p1_data = career_df[career_df["player_name"] == player1]
            p2_data = career_df[career_df["player_name"] == player2]
            
            if not p1_data.empty and not p2_data.empty:
                p1 = p1_data.iloc[0]
                p2 = p2_data.iloc[0]
                
                st.markdown("### Career Statistics Comparison")
                
                comp_stats = ["seasons", "games", "tot_pts", "tot_reb", "tot_ast", "hof_index"]
                comp_data = []
                
                for stat in comp_stats:
                    if stat in p1.index and stat in p2.index:
                        v1 = p1[stat] if not pd.isna(p1[stat]) else 0
                        v2 = p2[stat] if not pd.isna(p2[stat]) else 0
                        
                        comp_data.append({
                            "Statistic": stat.replace("tot_", "Total ").replace("_", " ").title(),
                            player1: f"{v1:.1f}" if stat == "hof_index" else int(v1),
                            player2: f"{v2:.1f}" if stat == "hof_index" else int(v2)
                        })
                
                if comp_data:
                    comp_df = pd.DataFrame(comp_data)
                    st.dataframe(comp_df, use_container_width=True)
                    
                    # Bar chart
                    st.markdown("### Visual Comparison")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    x = np.arange(len(comp_df))
                    width = 0.35
                    
                    vals1 = [float(str(v).replace(",", "")) for v in comp_df[player1]]
                    vals2 = [float(str(v).replace(",", "")) for v in comp_df[player2]]
                    
                    ax.bar(x - width/2, vals1, width, label=player1, alpha=0.8)
                    ax.bar(x + width/2, vals2, width, label=player2, alpha=0.8)
                    
                    ax.set_xticks(x)
                    ax.set_xticklabels(comp_df["Statistic"], rotation=45, ha="right")
                    ax.legend()
                    ax.set_ylabel("Value")
                    ax.set_title("Career Comparison")
                    st.pyplot(fig)

# TAB 5: TEAM TRENDS
with tabs[4]:
    st.subheader("Team trends")

    if "team" in season_df.columns and "year" in season_df.columns:
        team_list2 = sorted(season_df["team"].dropna().unique())
        if team_list2:
            sel_team2 = st.selectbox("Select team", team_list2)
            tdf2 = season_df[season_df["team"] == sel_team2]

            agg_cols2 = {}
            for c in ["pts", "reb", "ast"]:
                if c in tdf2.columns:
                    agg_cols2[c] = "sum"
            if "games" in tdf2.columns:
                agg_cols2["games"] = "sum"

            if agg_cols2:
                team_season_stats = tdf2.groupby("year").agg(agg_cols2).reset_index()
                st.markdown(f"### {sel_team2} – season totals")
                st.dataframe(team_season_stats)

                if "pts" in team_season_stats.columns:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(team_season_stats["year"], team_season_stats["pts"], marker="o")
                    ax.set_xlabel("Year")
                    ax.set_ylabel("Total points")
                    ax.set_title(f"{sel_team2} – total points per season")
                    st.pyplot(fig)

# TAB 6: HALL OF FAME EXPLORER
with tabs[5]:
    st.subheader("Hall of Fame Index Explorer (0-100)")

    st.markdown(
        """
**Hall of Fame Index** is a relative score (0-100) based on:
- Career seasons and games (longevity)
- Total points, rebounds, assists (production)
- Average team win% (team success)

Higher scores = more Hall-of-Fame-like profile.
"""
    )

    if career_df.empty:
        st.warning("Career table is empty")
    else:
        st.write(f"**Total players: {len(career_df):,}**")
        
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            if "seasons" in career_df.columns:
                max_s = int(career_df["seasons"].max()) if career_df["seasons"].notna().any() else 20
                min_seasons = st.slider("Minimum seasons", 0, max_s, 0)
            else:
                min_seasons = 0
        
        with col2:
            if "games" in career_df.columns:
                max_g = int(career_df["games"].max()) if career_df["games"].notna().any() else 1500
                min_games = st.slider("Minimum games", 0, max_g, 0, step=50)
            else:
                min_games = 0
        
        # Apply filters
        career_filtered = career_df.copy()
        if "seasons" in career_filtered.columns and min_seasons > 0:
            career_filtered = career_filtered[career_filtered["seasons"].fillna(0) >= min_seasons]
        if "games" in career_filtered.columns and min_games > 0:
            career_filtered = career_filtered[career_filtered["games"].fillna(0) >= min_games]
        
        if "hof_index" in career_filtered.columns:
            career_filtered = career_filtered.sort_values("hof_index", ascending=False)
        
        st.write(f"**Filtered: {len(career_filtered):,} players**")
        
        if len(career_filtered) > 0:
            top_n = st.slider("Show top N", 10, min(200, len(career_filtered)), min(50, len(career_filtered)), 10)

            st.markdown(f"### Top {top_n} players")
            
            display_cols = [c for c in ["player_name", "from_year", "to_year", "seasons", "games",
                                        "tot_pts", "tot_reb", "tot_ast", "avg_team_win_pct", "hof_index"] 
                           if c in career_filtered.columns]
            
            if display_cols:
                st.dataframe(career_filtered[display_cols].head(top_n), use_container_width=True)
        
        # Player inspector
        st.markdown("---")
        st.markdown("### Inspect player")
        
        if "player_name" in career_df.columns:
            all_players = sorted(career_df["player_name"].dropna().unique())
            
            if all_players:
                sel = st.selectbox(f"Select from {len(all_players):,} players", all_players)
                
                row = career_df[career_df["player_name"] == sel]
                if not row.empty:
                    r = row.iloc[0]
                    
                    st.markdown(f"## {sel}")
                    
                    c1, c2, c3 = st.columns(3)
                    
                    with c1:
                        st.markdown("**Career Overview**")
                        for col in ["from_year", "to_year", "seasons", "games"]:
                            if col in r.index and pd.notna(r[col]):
                                val = int(r[col]) if col != "games" else f"{int(r[col]):,}"
                                st.write(f"{col.replace('_', ' ').title()}: {val}")
                    
                    with c2:
                        st.markdown("**Career Totals**")
                        for col in ["tot_pts", "tot_reb", "tot_ast"]:
                            if col in r.index and pd.notna(r[col]):
                                st.write(f"{col.replace('tot_', '').upper()}: {int(r[col]):,}")
                    
                    with c3:
                        st.markdown("**Team Success**")
                        if "avg_team_win_pct" in r.index and pd.notna(r["avg_team_win_pct"]):
                            st.write(f"Avg Win%: {r['avg_team_win_pct']:.3f}")
                    
                    # HoF Index
                    if "hof_index" in r.index:
                        hof_idx = float(r["hof_index"])
                        st.markdown(f"### HoF Index: **{hof_idx:.1f} / 100**")
                        
                        if hof_idx >= 95:
                            verdict, color = "🏆 Elite", "green"
                        elif hof_idx >= 85:
                            verdict, color = "⭐ Strong HoF", "blue"
                        elif hof_idx >= 70:
                            verdict, color = "🎯 Borderline", "orange"
                        elif hof_idx >= 50:
                            verdict, color = "✅ Solid", "gray"
                        else:
                            verdict, color = "📊 Role player", "lightgray"
                        
                        st.markdown(f"**{verdict}**")
                        
                        fig, ax = plt.subplots(figsize=(8, 1.5))
                        ax.barh([0], [hof_idx], height=0.5, color=color)
                        ax.set_xlim(0, 100)
                        ax.set_ylim(-0.5, 0.5)
                        ax.set_yticks([])
                        ax.set_xlabel("HoF Index")
                        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.3)
                        ax.axvline(x=85, color='blue', linestyle='--', alpha=0.3)
                        ax.axvline(x=95, color='green', linestyle='--', alpha=0.3)
                        st.pyplot(fig)

    # Download
    st.markdown("---")
    def to_csv(df):
        return df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download career table",
        data=to_csv(career_df),
        file_name="career_with_hof_index.csv",
        mime="text/csv"
    )
