# ==================================================
# NBA FINAL PROJECT DASHBOARD (2004–05+)
# Uses 3 Kaggle datasets + HoF Index (0-100)
# ==================================================
import streamlit as st
import pandas as pd
import numpy as np
import os
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="NBA Analytics & Hall of Fame Explorer",
    page_icon="🏀",
    layout="wide"
)

sns.set()

SEASON_START_YEAR = 2004  # 2004–05 and later

# ==================================================
# 1. GENERIC HELPERS
# ==================================================
def download_first_csv(dataset_id, prefer_contains=None):
    """
    Download a Kaggle dataset via kagglehub and load one CSV file.
    """
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

    full_path = os.path.join(path, csv_name)
    return pd.read_csv(full_path)


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


def extract_season_start_from_string(season_value):
    """
    Convert things like '2004-05', '2004-2005', '2004' to 2004.
    """
    if pd.isna(season_value):
        return np.nan
    s = str(season_value)
    digits = "".join([ch for ch in s if ch.isdigit()])
    if len(digits) < 4:
        return np.nan
    try:
        year = int(digits[:4])
    except ValueError:
        return np.nan
    return year


def attach_season_start(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to create a 'season_start' column from 'season' or 'year'.
    """
    df = df.copy()
    season_col = None
    for name in ["season", "year", "season_year"]:
        if name in df.columns:
            season_col = name
            break

    if season_col is None:
        df["season_start"] = np.nan
        return df

    col_vals = df[season_col]
    if pd.api.types.is_numeric_dtype(col_vals):
        df["season_start"] = pd.to_numeric(col_vals, errors="coerce")
    else:
        df["season_start"] = col_vals.apply(extract_season_start_from_string)

    return df


def filter_to_modern_era(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep rows with season_start >= SEASON_START_YEAR.
    """
    if "season_start" not in df.columns:
        return df
    if not df["season_start"].notna().any():
        return df
    return df[df["season_start"] >= SEASON_START_YEAR].copy()


# ==================================================
# 2. LOAD DATASETS (Including Accolades)
# ==================================================
@st.cache_data
def load_players_season_stats() -> pd.DataFrame:
    """Dataset 1: drgilermo/nba-players-stats"""
    df = download_first_csv("drgilermo/nba-players-stats", prefer_contains="Seasons")
    df = normalize_columns(df)

    rename_map = {}
    for col in df.columns:
        if col in ["player", "player_name"]:
            rename_map[col] = "player_name"
        if col in ["tm", "team", "team_abbreviation"]:
            rename_map[col] = "team"
    df = df.rename(columns=rename_map)

    df = attach_season_start(df)
    df = filter_to_modern_era(df)

    return df


@st.cache_data
def load_traditional_box() -> pd.DataFrame:
    """Dataset 2: szymonjwiak/nba-traditional"""
    df = download_first_csv("szymonjwiak/nba-traditional")
    df = normalize_columns(df)

    rename_map = {}
    for col in df.columns:
        if col in ["player", "player_name"]:
            rename_map[col] = "player_name"
        if col in ["team_abbreviation", "tm", "team"]:
            rename_map[col] = "team"
    df = df.rename(columns=rename_map)

    df = attach_season_start(df)
    df = filter_to_modern_era(df)

    return df


@st.cache_data
def load_team_records() -> pd.DataFrame:
    """Dataset 3: boonpalipatana/nba-season-records-from-every-year"""
    df = download_first_csv("boonpalipatana/nba-season-records-from-every-year")
    df = normalize_columns(df)

    rename_map = {}
    for col in df.columns:
        if col in ["team", "franchise"]:
            rename_map[col] = "team"
        if "season" in col and col != "season_start":
            rename_map[col] = "season"
        if col in ["w", "wins"]:
            rename_map[col] = "wins"
        if col in ["l", "losses"]:
            rename_map[col] = "losses"
    df = df.rename(columns=rename_map)

    df = attach_season_start(df)
    df = filter_to_modern_era(df)

    if "wins" in df.columns and "losses" in df.columns:
        total_games = df["wins"].fillna(0) + df["losses"].fillna(0)
        df["win_pct"] = np.where(total_games > 0, df["wins"] / total_games, np.nan)

    return df


@st.cache_data
def load_accolades() -> pd.DataFrame:
    """Dataset 4: Comprehensive accolades (OPTIONAL)"""
    try:
        df = download_first_csv("ryanschubertds/all-nba-aba-players-bio-stats-accolades")
        df = normalize_columns(df)
        st.success(f"✅ Loaded accolades dataset: {len(df)} players")
        return df
    except Exception as e:
        st.warning(f"⚠️ Accolades dataset not available: {str(e)[:100]}")
        return pd.DataFrame()


# ==================================================
# 3. BUILD MERGED & CAREER TABLES + HOF INDEX WITH ACCOLADES
# ==================================================
@st.cache_data
def build_merged_data():
    players = load_players_season_stats()
    trad = load_traditional_box()
    teams = load_team_records()
    accolades = load_accolades()

    merged = players.copy()

    # merge team win% onto players
    if {"team", "season_start"}.issubset(merged.columns) and not teams.empty:
        team_small = teams[["team", "season_start", "win_pct"]].drop_duplicates()
        merged = merged.merge(
            team_small,
            on=["team", "season_start"],
            how="left",
            suffixes=("", "_team")
        )

    # Ensure we have a player_name column
    if "player_name" not in merged.columns:
        if "player" in merged.columns:
            merged = merged.rename(columns={"player": "player_name"})
        else:
            merged["player_name"] = "Unknown"

    # Build career-level aggregation
    agg_dict = {}
    if "season_start" in merged.columns:
        agg_dict["from_year"] = ("season_start", "min")
        agg_dict["to_year"] = ("season_start", "max")
        agg_dict["seasons"] = ("season_start", "nunique")
    if "g" in merged.columns:
        agg_dict["games"] = ("g", "sum")

    for stat in ["g", "mp", "pts", "trb", "ast", "stl", "blk", "tov"]:
        if stat in merged.columns:
            agg_dict[f"tot_{stat}"] = (stat, "sum")

    for stat in ["per", "ws", "bpm"]:
        if stat in merged.columns:
            agg_dict[f"avg_{stat}"] = (stat, "mean")

    if "win_pct" in merged.columns:
        agg_dict["avg_team_win_pct"] = ("win_pct", "mean")

    if agg_dict:
        career = merged.groupby("player_name").agg(**agg_dict).reset_index()
        st.write(f"**Career aggregation:** {len(career)} players, columns: {list(career.columns)[:10]}")
    else:
        career = pd.DataFrame(columns=["player_name"])
        st.warning("No aggregation dict created!")

    # ==================================================
    # PROCESS ACCOLADES (if available)
    # ==================================================
    accolade_cols_found = []
    
    if not accolades.empty:
        # Standardize player name
        if "player" in accolades.columns and "player_name" not in accolades.columns:
            accolades = accolades.rename(columns={"player": "player_name"})
        
        if "player_name" in accolades.columns:
            # Look for accolade columns (case-insensitive)
            accolade_map = {}
            
            for col in accolades.columns:
                col_lower = col.lower()
                
                # Major Awards (Highest Weight)
                if col_lower == "mvp":
                    accolade_map["mvp"] = col
                elif col_lower == "finals mvp":
                    accolade_map["finals_mvp"] = col
                elif col_lower == "championships":
                    accolade_map["championships"] = col
                
                # All-Star & Team Selections
                elif col_lower == "all star":
                    accolade_map["all_star"] = col
                elif col_lower == "all nba":
                    accolade_map["all_nba"] = col
                elif col_lower == "all aba":
                    accolade_map["all_aba"] = col
                elif col_lower == "all rookie":
                    accolade_map["all_rookie"] = col
                elif col_lower == "all defensive":
                    accolade_map["all_defensive"] = col
                
                # Defensive Awards
                elif col_lower == "dpoy":
                    accolade_map["dpoy"] = col
                
                # Individual Honors
                elif col_lower == "roy":
                    accolade_map["roy"] = col
                elif col_lower == "as mvp":
                    accolade_map["as_mvp"] = col
                elif col_lower == "cf mvp":
                    accolade_map["cf_mvp"] = col
                
                # Statistical Championships
                elif col_lower == "scoring champ":
                    accolade_map["scoring_champ"] = col
                elif col_lower == "ast champ":
                    accolade_map["ast_champ"] = col
                elif col_lower == "trb champ":
                    accolade_map["trb_champ"] = col
                elif col_lower == "stl champ":
                    accolade_map["stl_champ"] = col
                elif col_lower == "blk champ":
                    accolade_map["blk_champ"] = col
                
                # Role Player Awards
                elif col_lower == "most improved":
                    accolade_map["most_improved"] = col
                elif col_lower == "sixth man":
                    accolade_map["sixth_man"] = col
                
                # Legacy Honors
                elif col_lower == "nba 75 team":
                    accolade_map["nba_75"] = col
                elif col_lower == "aba all-time team":
                    accolade_map["aba_alltime"] = col
            
            if accolade_map:
                st.info(f"🏆 Found {len(accolade_map)} accolades: {accolade_map}")
                
                # Extract and aggregate
                accolades_subset = accolades[["player_name"] + list(accolade_map.values())].copy()
                
                # Show sample data
                st.write("**Accolade sample (first 5 rows):**")
                st.dataframe(accolades_subset.head())
                
                # Rename to standard names
                rename_dict = {v: k for k, v in accolade_map.items()}
                accolades_subset = accolades_subset.rename(columns=rename_dict)
                
                # Convert to numeric and sum by player
                for col in accolade_map.keys():
                    if col in accolades_subset.columns:
                        accolades_subset[col] = pd.to_numeric(accolades_subset[col], errors="coerce").fillna(0)
                
                accolades_agg = accolades_subset.groupby("player_name", as_index=False).sum()
                
                # Merge with career
                career = career.merge(accolades_agg, on="player_name", how="left")
                
                # Fill NaN with 0
                for col in accolade_map.keys():
                    if col in career.columns:
                        career[col] = career[col].fillna(0)
                        accolade_cols_found.append(col)
                
                st.success(f"✅ Merged accolades for {len(accolades_agg)} players")

    # --- HoF Index calculation (0-100 scale) WITH ACCOLADES ---
    def safe_log(x):
        return np.log1p(np.maximum(x, 0))

    career["hof_score_raw"] = 0.0

    # Base stats (production & longevity)
    if "tot_pts" in career.columns:
        career["hof_score_raw"] += 1.0 * safe_log(career["tot_pts"].fillna(0) / 1000.0)
        st.write(f"Added tot_pts: mean contribution = {(1.0 * safe_log(career['tot_pts'].fillna(0) / 1000.0)).mean():.2f}")
    if "tot_trb" in career.columns:
        career["hof_score_raw"] += 0.7 * safe_log(career["tot_trb"].fillna(0) / 500.0)
    if "tot_ast" in career.columns:
        career["hof_score_raw"] += 0.9 * safe_log(career["tot_ast"].fillna(0) / 500.0)
    if "games" in career.columns:
        career["hof_score_raw"] += 0.4 * safe_log(career["games"].fillna(0) / 200.0)
        st.write(f"Added games: mean contribution = {(0.4 * safe_log(career['games'].fillna(0) / 200.0)).mean():.2f}")
    if "avg_per" in career.columns:
        career["hof_score_raw"] += 0.8 * ((career["avg_per"] - 15.0) / 10.0)
    if "avg_ws" in career.columns:
        career["hof_score_raw"] += 0.8 * (career["avg_ws"] / 5.0)
    if "avg_team_win_pct" in career.columns:
        career["hof_score_raw"] += 0.6 * (career["avg_team_win_pct"] - 0.5) * 2.0

    # Accolades (heavily weighted if available)
    # Research-backed weights based on HoF voting patterns
    accolade_weights = {
        # Tier 1: MVP-Level Awards (Highest Impact)
        "mvp": 20.0,              # NBA MVP - Most important predictor
        "finals_mvp": 15.0,       # Finals MVP - 2nd most important
        "championships": 10.0,     # Championships/Rings - Critical for HoF
        
        # Tier 2: Elite Individual Recognition
        "dpoy": 6.0,              # Defensive Player of the Year
        "all_nba": 3.0,           # All-NBA selections (combined)
        "all_aba": 2.5,           # All-ABA selections (combined)
        
        # Tier 3: All-Star & Consistent Excellence
        "all_star": 2.0,          # All-Star selections
        "all_defensive": 1.5,     # All-Defensive selections
        "all_rookie": 0.8,        # All-Rookie team
        
        # Tier 4: Statistical Dominance
        "scoring_champ": 2.5,     # Scoring titles - high value
        "ast_champ": 2.0,         # Assist titles
        "trb_champ": 1.8,         # Rebound titles
        "blk_champ": 1.5,         # Block titles
        "stl_champ": 1.5,         # Steal titles
        
        # Tier 5: Secondary MVP Awards
        "as_mvp": 1.2,            # All-Star Game MVP
        "cf_mvp": 1.0,            # Conference Finals MVP
        "roy": 1.5,               # Rookie of the Year
        
        # Tier 6: Role Player Excellence
        "sixth_man": 0.8,         # Sixth Man Award
        "most_improved": 0.6,     # Most Improved Player
        
        # Tier 7: Legacy Honors (Recognition, not achievement)
        "nba_75": 3.0,            # NBA 75th Anniversary Team
        "aba_alltime": 2.0        # ABA All-Time Team
    }
    
    for accolade, weight in accolade_weights.items():
        if accolade in career.columns:
            career["hof_score_raw"] += weight * career[accolade]

    career["hof_score_raw"] = pd.to_numeric(career["hof_score_raw"], errors="coerce").fillna(0.0)

    if len(career) > 0:
        # Debug: Check score distribution
        st.write(f"**HoF Score Debug:** Min={career['hof_score_raw'].min():.2f}, Max={career['hof_score_raw'].max():.2f}, Mean={career['hof_score_raw'].mean():.2f}")
        st.write(f"Unique scores: {career['hof_score_raw'].nunique()}")
        
        # Convert to 0-100 percentile scale
        career["hof_index"] = (career["hof_score_raw"].rank(pct=True, method="average") * 100).round(1)
    else:
        career["hof_index"] = 0.0

    return merged, trad, teams, career, accolade_cols_found


merged_df, trad_df, teams_df, career_df, accolade_cols = build_merged_data()

# ==================================================
# 4. APP TABS
# ==================================================
st.title("🏀 NBA Analytics & Hall of Fame Explorer (2004–05+)")
st.caption("Data from three Kaggle datasets: player seasons, traditional stats, and team records.")

tabs = st.tabs([
    "Overview",
    "EDA (Players & Teams)",
    "Player Comparison",
    "Team Trends",
    "Hall of Fame Explorer",
    "Missing Data & Imputation",
    "Downloads"
])

# --------------------------------------------------
# TAB 1 – OVERVIEW
# --------------------------------------------------
with tabs[0]:
    st.subheader("Dataset overview")

    c1, c2 = st.columns([1.5, 1])

    with c1:
        st.markdown("**Season-level player stats:**")
        st.dataframe(merged_df.head(30))
        st.write(f"Rows: {merged_df.shape[0]:,}  |  Columns: {merged_df.shape[1]}")

    with c2:
        st.markdown("**Career-level table:**")
        st.dataframe(career_df.head(20))
        st.write(f"Players: {career_df.shape[0]:,}")

    st.markdown("**Columns (first 40):**")
    st.code(", ".join(merged_df.columns.tolist()[:40]))

    st.markdown("---")
    st.markdown("Created by **Aditya Sudarsan Anand** – CMSE 830 Final Project.")


# --------------------------------------------------
# TAB 2 – EDA
# --------------------------------------------------
with tabs[1]:
    st.subheader("Exploratory Data Analysis")

    num_cols = merged_df.select_dtypes(include=np.number).columns.tolist()

    colA, colB = st.columns(2)

    with colA:
        st.markdown("**Correlation heatmap**")
        if len(num_cols) >= 2:
            method = st.selectbox("Correlation method", ["pearson", "spearman"], index=0)
            corr = merged_df[num_cols].corr(method=method)
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
            ax.set_title(f"Correlation ({method})")
            st.pyplot(fig)
        else:
            st.info("Need at least two numeric columns.")

    with colB:
        st.markdown("**Stat distribution**")
        if num_cols:
            stat = st.selectbox("Select stat", num_cols, index=min(3, len(num_cols)-1))
            if "player_name" in merged_df.columns:
                players = sorted(merged_df["player_name"].dropna().unique())
                p_filter = st.selectbox("Filter by player (optional)", ["All"] + players)
            else:
                p_filter = "All"
            sub = merged_df if p_filter == "All" else merged_df[merged_df["player_name"] == p_filter]
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.histplot(sub[stat].dropna(), kde=True, bins=25, ax=ax)
            title = f"Distribution of {stat}" + ("" if p_filter == "All" else f" – {p_filter}")
            ax.set_title(title)
            st.pyplot(fig)
        else:
            st.info("No numeric stats detected.")

    st.markdown("---")
    st.subheader("Team-level win% distribution")
    if "win_pct" in teams_df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(teams_df["win_pct"].dropna(), kde=True, bins=25, ax=ax)
        ax.set_title("Distribution of team win percentage")
        st.pyplot(fig)
    else:
        st.info("Team win_pct column not found.")


# --------------------------------------------------
# TAB 3 – PLAYER COMPARISON
# --------------------------------------------------
with tabs[2]:
    st.subheader("Player comparison (head-to-head)")

    if "player_name" not in career_df.columns or career_df.empty:
        st.info("Career table is empty or missing player_name.")
    else:
        players = sorted(career_df["player_name"].dropna().unique())
        if len(players) < 2:
            st.info("Not enough players.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                p1 = st.selectbox("Player 1", players, key="cmp_p1")
            with c2:
                p2 = st.selectbox("Player 2", players, key="cmp_p2")

            stats_candidates = [
                c for c in [
                    "tot_pts", "tot_trb", "tot_ast", "tot_stl", "tot_blk",
                    "tot_tov", "games", "hof_index"
                ] if c in career_df.columns
            ]
            stats_selected = st.multiselect(
                "Metrics to compare",
                stats_candidates,
                default=stats_candidates
            )

            if stats_selected:
                sub = (
                    career_df.set_index("player_name")
                    .loc[[p1, p2], stats_selected]
                )

                fig, ax = plt.subplots(figsize=(8, 4))
                sub.T.plot(kind="bar", ax=ax)
                ax.set_title(f"{p1} vs {p2}")
                ax.set_ylabel("Value")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                st.pyplot(fig)

                def radar_plot(df_in: pd.DataFrame):
                    cats = list(df_in.columns)
                    norm = (df_in - df_in.min()) / (df_in.max() - df_in.min())
                    v1 = norm.iloc[0].fillna(0).values
                    v2 = norm.iloc[1].fillna(0).values
                    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
                    angles += angles[:1]
                    v1 = np.concatenate([v1, v1[:1]])
                    v2 = np.concatenate([v2, v2[:1]])

                    fig2 = plt.figure(figsize=(6, 6))
                    ax2 = plt.subplot(111, polar=True)
                    ax2.plot(angles, v1)
                    ax2.fill(angles, v1, alpha=0.1)
                    ax2.plot(angles, v2)
                    ax2.fill(angles, v2, alpha=0.1)
                    ax2.set_xticks(angles[:-1])
                    ax2.set_xticklabels(cats)
                    ax2.set_title("Normalized radar")
                    return fig2

                st.pyplot(radar_plot(sub))


# --------------------------------------------------
# TAB 4 – TEAM TRENDS
# --------------------------------------------------
with tabs[3]:
    st.subheader("Team trends over time (win%)")

    if teams_df.empty or "team" not in teams_df.columns or "season_start" not in teams_df.columns:
        st.info("Team records not available.")
    else:
        teams = sorted(teams_df["team"].dropna().unique())
        selected_team = st.selectbox("Select team", teams)
        sub = teams_df[teams_df["team"] == selected_team].sort_values("season_start")

        if "win_pct" in sub.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(sub["season_start"], sub["win_pct"], marker="o")
            ax.set_xlabel("Season start year")
            ax.set_ylabel("Win%")
            ax.set_title(f"{selected_team} win% over time")
            st.pyplot(fig)
        else:
            st.info("win_pct column missing.")


# --------------------------------------------------
# TAB 5 – HALL OF FAME EXPLORER
# --------------------------------------------------
with tabs[4]:
    st.subheader("Hall of Fame Index Explorer (0-100)")

    st.markdown(
        f"""
**Hall of Fame Index** (0-100) based on:

**Production & Longevity:**
- Career totals (points, rebounds, assists)
- Efficiency metrics (PER, WS, BPM if available)
- Longevity (seasons, games)
- Team success (average win%)

**Accolades (research-backed weights):**

**Tier 1 - MVP Awards:**
{f"- 🏆 MVP (20x weight)" if "mvp" in accolade_cols else ""}
{f"- 🏆 Finals MVP (15x weight)" if "finals_mvp" in accolade_cols else ""}
{f"- 💍 Championships (10x weight)" if "championships" in accolade_cols else ""}

**Tier 2 - Elite Recognition:**
{f"- 🛡️ DPOY (6x weight)" if "dpoy" in accolade_cols else ""}
{f"- ⭐ All-NBA (3x weight)" if "all_nba" in accolade_cols else ""}
{f"- ⭐ All-ABA (2.5x weight)" if "all_aba" in accolade_cols else ""}

**Tier 3 - Consistent Excellence:**
{f"- 🌟 All-Star (2x weight)" if "all_star" in accolade_cols else ""}
{f"- 🛡️ All-Defense (1.5x weight)" if "all_defensive" in accolade_cols else ""}

**Tier 4 - Statistical Dominance:**
{f"- 🎯 Scoring Titles (2.5x)" if "scoring_champ" in accolade_cols else ""}
{f"- 🎯 Assist/Rebound/Block/Steal Titles (2x/1.8x/1.5x/1.5x)" if any(c in accolade_cols for c in ["ast_champ", "trb_champ", "blk_champ", "stl_champ"]) else ""}

**Other Awards:**
{f"- 🆕 ROY, All-Rookie" if any(c in accolade_cols for c in ["roy", "all_rookie"]) else ""}
{f"- ⭐ All-Star Game MVP, Conf Finals MVP" if any(c in accolade_cols for c in ["as_mvp", "cf_mvp"]) else ""}
{f"- 📈 Sixth Man, Most Improved" if any(c in accolade_cols for c in ["sixth_man", "most_improved"]) else ""}
{f"- 🏅 NBA 75 Team, ABA All-Time Team" if any(c in accolade_cols for c in ["nba_75", "aba_alltime"]) else ""}

**Scale:**
- **95+**: Elite / Inner-circle HoF
- **85-94**: Strong HoF candidate
- **70-84**: Borderline HoF
- **50-69**: Solid career / All-Star
- **<50**: Role player
"""
    )

    if career_df.empty:
        st.info("Career table is empty.")
    else:
        st.write(f"**Total players: {len(career_df):,}**")

        # Filters
        col1, col2 = st.columns(2)

        with col1:
            min_seasons = st.slider("Minimum seasons", 0, 20, 0)

        with col2:
            min_games = st.slider("Minimum games", 0, 1500, 0, step=50)

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
            top_n = st.slider(
                "Show top N",
                10,
                min(200, len(career_filtered)),
                min(50, len(career_filtered)),
                10
            )

            st.markdown(f"### Top {top_n} players")

            display_cols = [c for c in [
                "player_name", "from_year", "to_year", "seasons", "games",
                "tot_pts", "tot_trb", "tot_ast", "avg_team_win_pct", "hof_index"
            ] if c in career_filtered.columns]

            st.dataframe(career_filtered[display_cols].head(top_n))

        # Player inspector - ALL PLAYERS
        st.markdown("---")
        st.markdown("### Inspect individual player")

        all_players = sorted(career_df["player_name"].dropna().unique())
        sel = st.selectbox(f"Select from ALL {len(all_players):,} players", all_players)
        row = career_df[career_df["player_name"] == sel]

        if not row.empty:
            r = row.iloc[0]

            st.markdown(f"## {sel}")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Career Overview**")
                if "from_year" in r.index and pd.notna(r["from_year"]):
                    st.write(f"From: {int(r['from_year'])}")
                if "to_year" in r.index and pd.notna(r["to_year"]):
                    st.write(f"To: {int(r['to_year'])}")
                if "seasons" in r.index and pd.notna(r["seasons"]):
                    st.write(f"Seasons: {int(r['seasons'])}")
                if "games" in r.index and pd.notna(r["games"]):
                    st.write(f"Games: {int(r['games']):,}")

            with col2:
                st.markdown("**Career Totals**")
                if "tot_pts" in r.index and pd.notna(r["tot_pts"]):
                    st.write(f"Points: {int(r['tot_pts']):,}")
                if "tot_trb" in r.index and pd.notna(r["tot_trb"]):
                    st.write(f"Rebounds: {int(r['tot_trb']):,}")
                if "tot_ast" in r.index and pd.notna(r["tot_ast"]):
                    st.write(f"Assists: {int(r['tot_ast']):,}")

            with col3:
                st.markdown("**Advanced Stats**")
                if "avg_per" in r.index and pd.notna(r["avg_per"]):
                    st.write(f"Avg PER: {r['avg_per']:.1f}")
                if "avg_ws" in r.index and pd.notna(r["avg_ws"]):
                    st.write(f"Avg WS: {r['avg_ws']:.1f}")
                if "avg_team_win_pct" in r.index and pd.notna(r["avg_team_win_pct"]):
                    st.write(f"Team win%: {r['avg_team_win_pct']:.3f}")
                
                # Show accolades if available
                st.markdown("**🏆 Accolades**")
                accolade_display = {
                    # Major Awards
                    "mvp": "🏆 MVP",
                    "finals_mvp": "🏆 Finals MVP", 
                    "championships": "💍 Championships",
                    
                    # All-Star & Teams
                    "all_star": "🌟 All-Star",
                    "all_nba": "⭐ All-NBA",
                    "all_aba": "⭐ All-ABA",
                    "all_rookie": "🆕 All-Rookie",
                    "all_defensive": "🛡️ All-Defense",
                    
                    # Defensive
                    "dpoy": "🛡️ DPOY",
                    
                    # Individual Honors
                    "roy": "🆕 ROY",
                    "as_mvp": "⭐ AS Game MVP",
                    "cf_mvp": "🏅 Conf Finals MVP",
                    
                    # Statistical Titles
                    "scoring_champ": "🎯 Scoring Titles",
                    "ast_champ": "🎯 Assist Titles",
                    "trb_champ": "🎯 Rebound Titles",
                    "stl_champ": "🎯 Steal Titles",
                    "blk_champ": "🎯 Block Titles",
                    
                    # Role Awards
                    "most_improved": "📈 MIP",
                    "sixth_man": "🔄 6th Man",
                    
                    # Legacy
                    "nba_75": "🏅 NBA 75",
                    "aba_alltime": "🏅 ABA All-Time"
                }
                
                accolades_shown = False
                for col, label in accolade_display.items():
                    if col in r.index and pd.notna(r[col]) and r[col] > 0:
                        st.write(f"{label}: {int(r[col])}")
                        accolades_shown = True
                
                if not accolades_shown:
                    st.caption("No accolade data available")

            # HoF Index
            hof_idx = float(r.get("hof_index", 0.0))
            st.markdown(f"### Hall of Fame Index: **{hof_idx:.1f} / 100**")

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
            ax.set_title(f"{sel} - HoF Index")
            ax.axvline(x=50, color='gray', linestyle='--', alpha=0.3, label='Avg')
            ax.axvline(x=85, color='blue', linestyle='--', alpha=0.3, label='Strong')
            ax.axvline(x=95, color='green', linestyle='--', alpha=0.3, label='Elite')
            ax.legend(loc='upper right')
            st.pyplot(fig)


# --------------------------------------------------
# TAB 6 – MISSING DATA
# --------------------------------------------------
with tabs[5]:
    st.subheader("Missing data & imputation")

    miss = merged_df.isna().mean().sort_values(ascending=False)
    st.markdown("**Fraction missing per column:**")
    st.dataframe(miss.to_frame("missing_fraction"))

    st.markdown("### Impute selected columns")
    cols_to_impute = st.multiselect(
        "Columns to impute",
        options=merged_df.columns.tolist(),
        default=[c for c in merged_df.select_dtypes(include=np.number).columns if merged_df[c].isna().any()]
    )
    strategy = st.selectbox("Imputation strategy", ["mean", "median", "mode", "ffill", "bfill"])
    n_preview = st.slider("Preview rows", 5, 50, 10, step=5)

    df_imp = merged_df.copy()
    for c in cols_to_impute:
        if strategy == "mean" and pd.api.types.is_numeric_dtype(df_imp[c]):
            df_imp[c] = df_imp[c].fillna(df_imp[c].mean())
        elif strategy == "median" and pd.api.types.is_numeric_dtype(df_imp[c]):
            df_imp[c] = df_imp[c].fillna(df_imp[c].median())
        elif strategy == "mode":
            if df_imp[c].mode().empty:
                df_imp[c] = df_imp[c].fillna(method="ffill").fillna(method="bfill")
            else:
                df_imp[c] = df_imp[c].fillna(df_imp[c].mode()[0])
        elif strategy == "ffill":
            df_imp[c] = df_imp[c].fillna(method="ffill")
        elif strategy == "bfill":
            df_imp[c] = df_imp[c].fillna(method="bfill")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Before**")
        st.dataframe(merged_df.head(n_preview))
    with c2:
        st.markdown("**After**")
        st.dataframe(df_imp.head(n_preview))


# --------------------------------------------------
# TAB 7 – DOWNLOADS
# --------------------------------------------------
with tabs[6]:
    st.subheader("Download cleaned tables")

    def to_csv_bytes(df_in: pd.DataFrame) -> bytes:
        return df_in.to_csv(index=False).encode("utf-8")

    st.markdown("**Player-season table**")
    st.download_button(
        label="⬇️ Download",
        data=to_csv_bytes(merged_df),
        file_name="player_season_merged.csv",
        mime="text/csv"
    )

    st.markdown("**Career table with HoF Index**")
    st.download_button(
        label="⬇️ Download",
        data=to_csv_bytes(career_df),
        file_name="career_with_hof_index.csv",
        mime="text/csv"
    )

    st.markdown("**Team records**")
    st.download_button(
        label="⬇️ Download",
        data=to_csv_bytes(teams_df),
        file_name="team_records.csv",
        mime="text/csv"
    )
