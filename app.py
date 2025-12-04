# ==================================================
# CMSE 830 FINAL PROJECT - ENHANCED WITH ACCOLADES
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

st.title("🏀 NBA Analytics & Hall of Fame Index Dashboard (with Accolades)")

st.markdown(
    """
This app combines **multiple NBA datasets** including **awards and accolades** to build:

- Season & team-level exploration  
- Player comparison and trends  
- A **Hall of Fame Index (0–100)** that scores every player based on:
  - Longevity, production, and team success
  - **Awards: MVP, All-Star, All-NBA, Championships, Finals MVP**

_Data sources: Kaggle datasets for player stats, team records, and awards._
"""
)

# --------------------------------------------------
# HELPER: generic Kaggle loader
# --------------------------------------------------
def load_kaggle_csv(dataset_id: str, prefer_contains=None) -> pd.DataFrame:
    """Download a Kaggle dataset and return the first (or preferred) CSV."""
    path = kagglehub.dataset_download(dataset_id)
    files = [f for f in os.listdir(path) if f.lower().endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"No CSV files found in Kaggle dataset {dataset_id}")
    
    csv_name = files[0]
    if prefer_contains:
        for f in files:
            if prefer_contains.lower() in f.lower():
                csv_name = f
                break
    
    csv_path = os.path.join(path, csv_name)
    return pd.read_csv(csv_path)


@st.cache_data(show_spinner=True)
def load_all_raw():
    """Load player stats, boxscores, team records, and COMPREHENSIVE awards data."""
    players_raw = load_kaggle_csv("drgilermo/nba-players-stats")
    boxscores_raw = load_kaggle_csv("szymonjwiak/nba-traditional")
    seasons_raw = load_kaggle_csv("boonpalipatana/nba-season-records-from-every-year")
    
    # Load COMPREHENSIVE accolades dataset (has everything!)
    try:
        accolades_raw = load_kaggle_csv("ryanschubertds/all-nba-aba-players-bio-stats-accolades")
        st.success("✅ Loaded comprehensive accolades dataset (MVP, Championships, Finals MVP, DPOY, All-NBA, All-Defense, etc.)")
    except Exception as e:
        st.warning(f"⚠️ Comprehensive accolades dataset not found: {e}")
        accolades_raw = pd.DataFrame()
    
    # Load MVP voting dataset as backup
    try:
        mvp_raw = load_kaggle_csv("robertsunderhaft/nba-player-season-statistics-with-mvp-win-share")
    except:
        mvp_raw = pd.DataFrame()
    
    # Load All-Star data
    try:
        allstar_raw = load_kaggle_csv("ahmedbendaly/nba-all-star-game-data", prefer_contains="players")
    except:
        allstar_raw = pd.DataFrame()

    return players_raw, boxscores_raw, seasons_raw, accolades_raw, mvp_raw, allstar_raw


# --------------------------------------------------
# DATA WRANGLING
# --------------------------------------------------
@st.cache_data(show_spinner=True)
def build_clean_tables():
    players_raw, boxscores_raw, seasons_raw, accolades_raw, mvp_raw, allstar_raw = load_all_raw()

    # Normalize column names
    players = players_raw.copy()
    players.columns = players.columns.str.strip().str.lower()

    box = boxscores_raw.copy()
    box.columns = box.columns.str.strip().str.lower()

    seasons = seasons_raw.copy()
    seasons.columns = seasons.columns.str.strip().str.lower()

    # Career-level table
    career = players.copy()

    if "player" in career.columns and "player_name" not in career.columns:
        career = career.rename(columns={"player": "player_name"})
    if "from" in career.columns:
        career = career.rename(columns={"from": "from_year"})
    if "to" in career.columns:
        career = career.rename(columns={"to": "to_year"})
    if "yrs" in career.columns:
        career = career.rename(columns={"yrs": "seasons"})
    if "g" in career.columns and "games" not in career.columns:
        career = career.rename(columns={"g": "games"})

    numeric_cols = ["from_year", "to_year", "seasons", "games", "tot_pts", "tot_trb", 
                    "tot_ast", "tot_stl", "tot_blk", "ws", "bpm", "vorp"]
    for col in numeric_cols:
        if col in career.columns:
            career[col] = pd.to_numeric(career[col], errors="coerce")

    # Season-level from boxscores
    rename_box = {}
    if "season" in box.columns:
        rename_box["season"] = "season"
    elif "season_id" in box.columns:
        rename_box["season_id"] = "season"
    
    if "player_name" in box.columns:
        rename_box["player_name"] = "player_name"
    elif "player" in box.columns:
        rename_box["player"] = "player_name"

    if "team_abbreviation" in box.columns:
        rename_box["team_abbreviation"] = "team"
    elif "team" in box.columns:
        rename_box["team"] = "team"

    for stat in ["pts", "reb", "ast", "stl", "blk", "tov", "min"]:
        if stat in box.columns:
            rename_box[stat] = stat
    
    if "game_id" in box.columns:
        rename_box["game_id"] = "game_id"

    box = box.rename(columns=rename_box)
    keep_cols = [c for c in ["season", "player_name", "team", "game_id", "pts", "reb", "ast", "stl", "blk"] if c in box.columns]
    box = box[keep_cols]
    
    if "season" in box.columns and "player_name" in box.columns:
        box = box.dropna(subset=["season", "player_name"])

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
        
        box["season_start"] = box["season"].apply(season_to_year)
        box = box.dropna(subset=["season_start"])
        box["season_start"] = box["season_start"].astype(int)
        box = box[box["season_start"] >= 2005]

    # Aggregate to player-season
    agg_dict = {}
    for c in ["pts", "reb", "ast", "stl", "blk"]:
        if c in box.columns:
            agg_dict[c] = "sum"
    if "game_id" in box.columns:
        agg_dict["game_id"] = pd.Series.nunique

    if agg_dict and "season_start" in box.columns:
        season_player = (
            box.groupby(["season_start", "player_name", "team"], as_index=False)
            .agg(agg_dict)
            .rename(columns={"season_start": "year", "game_id": "games"})
        )
    else:
        season_player = pd.DataFrame(columns=["year", "player_name", "team"])

    # Team records
    if "season" in seasons.columns:
        seasons = seasons.rename(columns={"season": "year"})
    if "team_name" in seasons.columns:
        seasons = seasons.rename(columns={"team_name": "team"})
    if "win%" in seasons.columns:
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
    
    for stat in ["pts", "reb", "ast", "stl", "blk"]:
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
        career_extra = career.drop_duplicates(subset=["player_name"])
        career_all = career_from_box.merge(career_extra, on="player_name", how="left", suffixes=("", "_orig"))
    else:
        career_all = career_from_box.copy()

    # ==================================================
    # PROCESS COMPREHENSIVE ACCOLADES DATA
    # ==================================================
    awards_career = pd.DataFrame()
    
    if not accolades_raw.empty:
        accolades = accolades_raw.copy()
        accolades.columns = accolades.columns.str.strip().str.lower()
        
        # Standardize player name column
        name_cols = ["player", "player_name", "name"]
        for col in name_cols:
            if col in accolades.columns:
                accolades = accolades.rename(columns={col: "player_name"})
                break
        
        if "player_name" in accolades.columns:
            # Map expected accolade column names
            accolade_mappings = {
                # Championships
                "championships": ["championships", "rings", "titles", "champion"],
                # MVPs
                "mvp": ["mvp", "mvps", "league_mvp", "season_mvp"],
                "finals_mvp": ["finals_mvp", "fmvp", "finals_mvps"],
                # All-NBA
                "all_nba_first": ["all_nba_first", "all-nba_first", "all_nba_1st"],
                "all_nba_second": ["all_nba_second", "all-nba_second", "all_nba_2nd"],
                "all_nba_third": ["all_nba_third", "all-nba_third", "all_nba_3rd"],
                # All-Star
                "all_star": ["all_star", "allstar", "all_stars", "all-star"],
                # Defense
                "dpoy": ["dpoy", "defensive_player_of_the_year", "dpoy_awards"],
                "all_defensive_first": ["all_defensive_first", "all-defensive_first", "all_def_1st"],
                "all_defensive_second": ["all_defensive_second", "all-defensive_second", "all_def_2nd"],
                # Other
                "roy": ["roy", "rookie_of_the_year"],
                "scoring_titles": ["scoring_champion", "scoring_titles", "scoring_leader"]
            }
            
            # Find and standardize columns
            standardized_cols = {}
            for standard_name, possible_names in accolade_mappings.items():
                for possible in possible_names:
                    if possible in accolades.columns:
                        standardized_cols[possible] = standard_name
                        break
            
            accolades = accolades.rename(columns=standardized_cols)
            
            # Aggregate accolades per player
            agg_dict = {}
            for accolade in accolade_mappings.keys():
                if accolade in accolades.columns:
                    agg_dict[accolade] = "sum"
            
            if agg_dict:
                awards_career = accolades.groupby("player_name").agg(**agg_dict).reset_index()
                
                # Combine All-NBA into total count
                allnba_cols = ["all_nba_first", "all_nba_second", "all_nba_third"]
                existing_allnba = [c for c in allnba_cols if c in awards_career.columns]
                if existing_allnba:
                    awards_career["all_nba_total"] = awards_career[existing_allnba].sum(axis=1)
                
                # Combine All-Defense into total count
                alldef_cols = ["all_defensive_first", "all_defensive_second"]
                existing_alldef = [c for c in alldef_cols if c in awards_career.columns]
                if existing_alldef:
                    awards_career["all_defensive_total"] = awards_career[existing_alldef].sum(axis=1)
    
    # Fallback to MVP dataset if comprehensive one failed
    elif not mvp_raw.empty:
        mvp = mvp_raw.copy()
        mvp.columns = mvp.columns.str.strip().str.lower()
        
        if "player" in mvp.columns:
            mvp = mvp.rename(columns={"player": "player_name"})
        
        if "player_name" in mvp.columns:
            mvp_col = None
            for col in ["mvp", "is_mvp", "mvp_winner", "award_share"]:
                if col in mvp.columns:
                    mvp_col = col
                    break
            
            if mvp_col:
                if mvp_col == "award_share":
                    mvp_winners = mvp[pd.to_numeric(mvp[mvp_col], errors="coerce") == 1.0]
                else:
                    mvp_winners = mvp[mvp[mvp_col] == True]
                
                if not mvp_winners.empty:
                    awards_career = mvp_winners.groupby("player_name").size().reset_index(name="mvp")
    
    # Process All-Star data
    if not allstar_raw.empty:
        allstar = allstar_raw.copy()
        allstar.columns = allstar.columns.str.strip().str.lower()
        
        if "player_name" in allstar.columns:
            allstar_counts = allstar.groupby("player_name").size().reset_index(name="all_star")
            
            if awards_career.empty:
                awards_career = allstar_counts
            else:
                awards_career = awards_career.merge(allstar_counts, on="player_name", how="outer")
    
    # Merge awards into career data
    if not awards_career.empty and "player_name" in career_all.columns:
        career_all = career_all.merge(awards_career, on="player_name", how="left")
        
        # Fill NaN with 0 for all award columns
        award_cols = [c for c in awards_career.columns if c != "player_name"]
        for col in award_cols:
            if col in career_all.columns:
                career_all[col] = career_all[col].fillna(0)

    return season_merged, team_seasons, career_all


season_df, team_df, career_df = build_clean_tables()

# --------------------------------------------------
# ENHANCED HOF INDEX (0–100) WITH ALL ACCOLADES
# --------------------------------------------------
def add_hof_index_with_accolades(career: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced HoF Index with COMPREHENSIVE accolades.
    Weights based on research from medium.com/@dwang22:
    - Finals MVP > Championships > DPOY > All-NBA 1st > All-Star > All-Defense 1st > All-Rookie 1st
    """
    df = career.copy()

    # Base features (production + longevity)
    base_features = [c for c in ["seasons", "games", "tot_pts", "tot_reb", "tot_ast", "avg_team_win_pct"] 
                     if c in df.columns]
    
    # Accolade features with research-backed weights
    accolade_weights = {
        "mvp": 15.0,                      # NBA MVP - Most important!
        "finals_mvp": 12.0,               # Finals MVP - 2nd most important
        "championships": 8.0,              # Championships/Rings
        "dpoy": 5.0,                      # Defensive Player of the Year
        "all_nba_first": 4.0,             # All-NBA First Team
        "all_nba_total": 2.5,             # All-NBA (any team)
        "all_star": 2.0,                  # All-Star selections
        "all_defensive_first": 2.0,       # All-Defense First Team
        "all_defensive_total": 1.0,       # All-Defense (any team)
        "roy": 1.5,                       # Rookie of the Year
        "scoring_titles": 1.5             # Scoring Champion
    }
    
    accolade_features = [c for c in accolade_weights.keys() if c in df.columns]
    all_features = base_features + accolade_features
    
    if not all_features:
        df["hof_index"] = 0.0
        return df

    X = df[all_features].fillna(0).astype(float)
    
    # Apply weights to accolades
    X_weighted = X.copy()
    for accolade, weight in accolade_weights.items():
        if accolade in X_weighted.columns:
            X_weighted[accolade] = X_weighted[accolade] * weight
    
    # Z-score normalization
    mean = X_weighted.mean()
    std = X_weighted.std(ddof=0) + 1e-10
    z = (X_weighted - mean) / std
    z = z.fillna(0.0)

    raw_score = z.sum(axis=1)
    
    # Convert to 0-100 percentile
    ranks = raw_score.rank(method="average", pct=True)
    df["hof_index"] = (ranks * 100).round(1)

    return df


career_df = add_hof_index_with_accolades(career_df)

# Check what accolade columns we actually have
all_possible_accolades = ["mvp", "finals_mvp", "championships", "dpoy", "all_nba_first", 
                          "all_nba_second", "all_nba_third", "all_nba_total", "all_star",
                          "all_defensive_first", "all_defensive_second", "all_defensive_total",
                          "roy", "scoring_titles"]
accolade_cols_available = [c for c in all_possible_accolades if c in career_df.columns]

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
    
    # Summary metrics
    st.markdown("### 📊 Dataset Summary")
    col_a, col_b, col_c, col_d = st.columns(4)
    
    with col_a:
        st.metric("Total Players", f"{len(career_df):,}")
    
    with col_b:
        unique_seasons = season_df["player_name"].dropna().nunique() if "player_name" in season_df.columns else 0
        st.metric("Players (Season Data)", f"{unique_seasons:,}")
    
    with col_c:
        total_seasons = len(season_df) if not season_df.empty else 0
        st.metric("Player-Seasons", f"{total_seasons:,}")
    
    with col_d:
        year_range = ""
        if "year" in season_df.columns and not season_df.empty:
            min_year = int(season_df["year"].min())
            max_year = int(season_df["year"].max())
            year_range = f"{min_year}-{max_year}"
        st.metric("Year Range", year_range if year_range else "N/A")
    
    # Accolades status
    st.markdown("### 🏆 Accolades Data Status")
    if accolade_cols_available:
        st.success(f"✅ **{len(accolade_cols_available)} accolades included:**")
        col_display = {
            "mvp": "🏆 MVP",
            "finals_mvp": "🏆 Finals MVP",
            "championships": "💍 Championships/Rings",
            "dpoy": "🛡️ DPOY",
            "all_nba_first": "⭐ All-NBA 1st",
            "all_nba_second": "⭐ All-NBA 2nd",
            "all_nba_third": "⭐ All-NBA 3rd",
            "all_nba_total": "⭐ All-NBA Total",
            "all_star": "🌟 All-Star",
            "all_defensive_first": "🛡️ All-Defense 1st",
            "all_defensive_second": "🛡️ All-Defense 2nd",
            "all_defensive_total": "🛡️ All-Defense Total",
            "roy": "🆕 ROY",
            "scoring_titles": "🎯 Scoring Titles"
        }
        displayed = [col_display.get(c, c) for c in accolade_cols_available]
        st.write(", ".join(displayed))
    else:
        st.warning("⚠️ No accolades data found. HoF Index uses only stats.")
    
    st.markdown("---")
    
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
                comp_stats += [c for c in accolade_cols_available if c in p1.index and c in p2.index]
                
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
    st.subheader("Hall of Fame Index Explorer (0-100) - WITH ACCOLADES")

    st.markdown(
        f"""
**Enhanced Hall of Fame Index** - Research-backed weights for all accolades:

**Production & Longevity:**
- Career seasons, games, points, rebounds, assists
- Average team win%

**Accolades (weights based on HoF research):**
{f"- 🏆 **MVP** (15x weight) - Most important HoF predictor" if "mvp" in accolade_cols_available else "- ❌ MVP (not available)"}
{f"- 🏆 **Finals MVP** (12x weight) - 2nd most important" if "finals_mvp" in accolade_cols_available else "- ❌ Finals MVP (not available)"}
{f"- 💍 **Championships/Rings** (8x weight)" if "championships" in accolade_cols_available else "- ❌ Championships (not available)"}
{f"- 🛡️ **DPOY** (5x weight)" if "dpoy" in accolade_cols_available else "- ❌ DPOY (not available)"}
{f"- ⭐ **All-NBA 1st Team** (4x weight)" if "all_nba_first" in accolade_cols_available else "- ❌ All-NBA 1st (not available)"}
{f"- ⭐ **All-NBA Total** (2.5x weight)" if "all_nba_total" in accolade_cols_available else ""}
{f"- 🌟 **All-Star** (2x weight)" if "all_star" in accolade_cols_available else "- ❌ All-Star (not available)"}
{f"- 🛡️ **All-Defense** (2x/1x weight)" if any(c in accolade_cols_available for c in ["all_defensive_first", "all_defensive_total"]) else "- ❌ All-Defense (not available)"}
{f"- 🆕 **ROY** (1.5x weight)" if "roy" in accolade_cols_available else ""}
{f"- 🎯 **Scoring Titles** (1.5x weight)" if "scoring_titles" in accolade_cols_available else ""}

_Research source: Logistic regression analysis of actual HoF inductees_
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
                                        "tot_pts", "tot_reb", "tot_ast"] + accolade_cols_available + ["hof_index"]
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
                        st.markdown("**Accolades** 🏆")
                        for col in accolade_cols_available:
                            if col in r.index and pd.notna(r[col]) and r[col] > 0:
                                label = col.replace("_", " ").title()
                                st.write(f"{label}: {int(r[col])}")
                        
                        if "avg_team_win_pct" in r.index and pd.notna(r["avg_team_win_pct"]):
                            st.write(f"Avg Win%: {r['avg_team_win_pct']:.3f}")
                    
                    # HoF Index
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
                        
                        fig, ax = plt.subplots(figsize=(8, 1.5))
                        ax.barh([0], [hof_idx], height=0.5, color=color)
                        ax.set_xlim(0, 100)
                        ax.set_ylim(-0.5, 0.5)
                        ax.set_yticks([])
                        ax.set_xlabel("HoF Index (0-100)")
                        ax.set_title(f"{sel} - Enhanced HoF Index (with accolades)")
                        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.3, label='Avg')
                        ax.axvline(x=85, color='blue', linestyle='--', alpha=0.3, label='Strong')
                        ax.axvline(x=95, color='green', linestyle='--', alpha=0.3, label='Elite')
                        ax.legend(loc='upper right', fontsize=8)
                        st.pyplot(fig)

    # Download
    st.markdown("---")
    def to_csv(df):
        return df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download career table with accolades & HoF Index",
        data=to_csv(career_df),
        file_name="career_with_accolades_hof_index.csv",
        mime="text/csv"
    )
