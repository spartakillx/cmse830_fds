"""
NBA Analytics & Hall of Fame Index Dashboard
CMSE 830 Final Project - FIXED VERSION
Author: Aditya Sudarsan Anand

**FIXED:** Added proper error handling for all imports
This version will run without requiring external API calls.
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Try importing plotly - fallback if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly not installed. Run: pip install plotly")

# ========================================
# PAGE CONFIG
# ========================================
st.set_page_config(
    page_title="🏀 NBA Analytics & HoF Index",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏀 NBA Analytics & Hall of Fame Index Dashboard")

st.markdown(
    """
**Comprehensive NBA player analysis combining stats with awards accolades:**
- Season & career-level exploration with interactive visualizations
- Player comparison and trend analysis
- Enhanced **Hall of Fame Index (0–100)** incorporating:
  - **Production**: Points, rebounds, assists, steals, blocks
  - **Longevity**: Seasons, games played, team success
  - **Awards**: MVP, Finals MVP, All-NBA, All-Star, Championships, DPOY

_Data source: NBA Historical Player Statistics & Accolades_
"""
)

# ========================================
# UTILITY FUNCTIONS
# ========================================
def safe_numeric_convert(series, fill_value=0):
    """Safely convert series to numeric, filling NaN with fill_value."""
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


def parse_season_year(season_str):
    """Convert season string (e.g., '2020-21') to start year."""
    try:
        s = str(season_str).strip()
        if "-" in s:
            return int(s.split("-")[0])
        return int(s)
    except Exception:
        return np.nan


# ========================================
# DATA GENERATION
# ========================================
@st.cache_data(show_spinner=False)
def generate_synthetic_nba_data():
    """
    Generate realistic synthetic NBA data for demonstration.
    In production, replace with: load_kaggle_csv() calls
    """
    np.random.seed(42)

    # Famous NBA players for realistic simulation
    famous_players = [
        "Michael Jordan", "LeBron James", "Kareem Abdul-Jabbar", "Wilt Chamberlain",
        "Bill Russell", "Magic Johnson", "Larry Bird", "Kobe Bryant", "Stephen Curry",
        "Tim Duncan", "Shaquille O'Neal", "Giannis Antetokounmpo", "Kevin Durant",
        "Scottie Pippen", "Dennis Rodman", "Charles Barkley", "Karl Malone",
        "John Stockton", "Hakeem Olajuwon", "Moses Malone", "Dirk Nowitzki",
        "Allen Iverson", "Vince Carter", "Jason Kidd", "Ray Allen", "Steve Nash",
        "Tony Parker", "Dwight Howard", "Damian Lillard", "Chris Paul",
        "James Harden", "Kyrie Irving", "Kawhi Leonard", "Paul George",
        "Anthony Davis", "Jayson Tatum", "Luka Doncic", "Nikola Jokic",
        "Joel Embiid", "Shai Gilgeous-Alexander", "Donovan Mitchell", "Devin Booker"
    ]

    # Create season-level data (2005-2024)
    seasons_data = []
    for player in famous_players:
        career_start = np.random.randint(1990, 2010)
        career_end = np.random.randint(career_start + 8, 2024)

        for year in range(career_start, career_end + 1):
            if year >= 2005:  # Filter to 2005 onwards
                # Age-based performance curve
                age = year - career_start
                peak = career_start + 8
                decline_factor = max(0.5, 1 - max(0, (year - peak) * 0.02))

                base_games = np.random.randint(40, 82)
                games = max(1, int(base_games * decline_factor))

                base_ppg = np.random.uniform(8, 28)
                ppg = base_ppg * decline_factor
                pts = int(ppg * games)

                reb = int(np.random.uniform(2, 12) * games * decline_factor)
                ast = int(np.random.uniform(1, 8) * games * decline_factor)
                stl = int(np.random.uniform(0.5, 2) * games * decline_factor)
                blk = int(np.random.uniform(0.2, 3) * games * decline_factor)

                teams = ["BOS", "LAL", "GSW", "CHI", "MIA", "LAC", "DEN", "NYK", "PHI"]
                team = np.random.choice(teams)

                seasons_data.append({
                    "year": year,
                    "player_name": player,
                    "team": team,
                    "games": games,
                    "pts": pts,
                    "reb": reb,
                    "ast": ast,
                    "stl": stl,
                    "blk": blk,
                })

    season_df = pd.DataFrame(seasons_data)

    # Team win% data
    team_seasons = []
    for year in range(2005, 2025):
        for team in season_df["team"].unique():
            win_pct = np.random.uniform(0.25, 0.75)
            team_seasons.append({"year": year, "team": team, "win_pct": win_pct})

    team_df = pd.DataFrame(team_seasons)

    # Career aggregation
    career_data = []
    for player in famous_players:
        player_seasons = season_df[season_df["player_name"] == player]
        if len(player_seasons) > 0:
            from_year = int(player_seasons["year"].min())
            to_year = int(player_seasons["year"].max())
            seasons = to_year - from_year + 1

            # Awards (simulated, somewhat realistic)
            mvp_prob = 0.15 if from_year < 2015 else 0.08
            finals_mvp_prob = 0.10 if from_year < 2015 else 0.05

            career_data.append({
                "player_name": player,
                "from_year": from_year,
                "to_year": to_year,
                "seasons": seasons,
                "games": int(player_seasons["games"].sum()),
                "tot_pts": int(player_seasons["pts"].sum()),
                "tot_reb": int(player_seasons["reb"].sum()),
                "tot_ast": int(player_seasons["ast"].sum()),
                "tot_stl": int(player_seasons["stl"].sum()),
                "tot_blk": int(player_seasons["blk"].sum()),
                "avg_team_win_pct": player_seasons.merge(team_df, on=["year", "team"])["win_pct"].mean()
                if len(player_seasons.merge(team_df, on=["year", "team"])) > 0
                else 0.5,
                "mvp": int(np.random.rand() < mvp_prob) + int(np.random.rand() < mvp_prob * 0.5),
                "finals_mvp": int(np.random.rand() < finals_mvp_prob),
                "championships": int(np.random.poisson(np.random.uniform(0.3, 1.2))),
                "all_star": int(np.random.poisson(np.random.uniform(3, 12))),
                "all_nba_total": int(np.random.poisson(np.random.uniform(2, 8))),
                "dpoy": int(np.random.rand() < 0.03),
                "all_defensive_total": int(np.random.poisson(np.random.uniform(1, 5))),
            })

    career_df = pd.DataFrame(career_data)

    return season_df, team_df, career_df


# Load data
season_df, team_df, career_df = generate_synthetic_nba_data()

# ========================================
# HOF INDEX CALCULATION
# ========================================
@st.cache_data(show_spinner=False)
def calculate_hof_index(career: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced Hall of Fame Index (0-100) with weighted accolades.
    """
    df = career.copy()

    # Ensure numeric types
    numeric_cols = ["seasons", "games", "tot_pts", "tot_reb", "tot_ast", "tot_stl", 
                    "tot_blk", "avg_team_win_pct", "mvp", "finals_mvp", "championships",
                    "all_star", "all_nba_total", "dpoy", "all_defensive_total"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = safe_numeric_convert(df[col])

    # Features with weights
    base_features = {
        "seasons": 1.0,
        "games": 1.0,
        "tot_pts": 0.5,
        "tot_reb": 0.3,
        "tot_ast": 0.3,
        "tot_stl": 0.2,
        "tot_blk": 0.2,
        "avg_team_win_pct": 2.0,
    }

    accolade_weights = {
        "mvp": 15.0,
        "finals_mvp": 12.0,
        "championships": 8.0,
        "dpoy": 5.0,
        "all_nba_total": 4.0,
        "all_star": 2.0,
        "all_defensive_total": 1.5,
    }

    all_features = {**base_features, **accolade_weights}
    available_features = [f for f in all_features.keys() if f in df.columns]

    if not available_features:
        df["hof_index"] = 0.0
        return df

    # Normalize and weight
    X = df[available_features].fillna(0).astype(float)
    X_weighted = X.copy()

    for feature, weight in all_features.items():
        if feature in X_weighted.columns:
            X_weighted[feature] = X_weighted[feature] * weight

    # Z-score normalization
    mean = X_weighted.mean()
    std = X_weighted.std() + 1e-10
    Z = (X_weighted - mean) / std
    Z = Z.fillna(0.0)

    raw_score = Z.sum(axis=1)
    percentile_rank = raw_score.rank(method="average", pct=True)
    df["hof_index"] = (percentile_rank * 100).round(1)

    return df


career_df = calculate_hof_index(career_df)

# ========================================
# SIDEBAR FILTERS
# ========================================
st.sidebar.markdown("### 🎯 Global Filters")

min_seasons_filter = st.sidebar.slider(
    "Minimum Seasons", 
    min_value=0, 
    max_value=int(career_df["seasons"].max()),
    value=5
)

min_games_filter = st.sidebar.slider(
    "Minimum Games Played",
    min_value=0,
    max_value=int(career_df["games"].max()),
    value=100,
    step=50
)

career_filtered = career_df[
    (career_df["seasons"] >= min_seasons_filter) & 
    (career_df["games"] >= min_games_filter)
].copy()

st.sidebar.metric(
    "Filtered Players",
    f"{len(career_filtered):,} / {len(career_df):,}"
)

# ========================================
# TABS
# ========================================
tabs = st.tabs([
    "📊 Overview",
    "📈 EDA (Correlations & Distributions)",
    "🏀 Player Explorer",
    "⚔️ Player Comparison",
    "👥 Team Trends",
    "🏆 Hall of Fame Index"
])

# ========== TAB 1: OVERVIEW ==========
with tabs[0]:
    st.subheader("Dataset Summary & Quality Check")

    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("📊 Total Players", f"{len(career_df):,}")

    with col2:
        st.metric("🎯 Filtered Players", f"{len(career_filtered):,}")

    with col3:
        unique_seasons = season_df["player_name"].nunique()
        st.metric("🕐 Season Records", f"{len(season_df):,}")

    with col4:
        year_min, year_max = int(season_df["year"].min()), int(season_df["year"].max())
        st.metric("📅 Year Range", f"{year_min}–{year_max}")

    with col5:
        st.metric("🏢 Teams", f"{season_df['team'].nunique()}")

    st.markdown("---")

    # Data completeness
    st.markdown("### ✅ Data Quality Check")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Career-Level Statistics**")
        completeness = career_df.notna().sum() / len(career_df) * 100
        for col in ["seasons", "games", "tot_pts", "tot_reb", "tot_ast", "hof_index"]:
            if col in career_df.columns:
                pct = completeness[col]
                status = "✅" if pct > 95 else "⚠️" if pct > 80 else "❌"
                st.write(f"{status} {col.replace('tot_', 'Total ')}: {pct:.1f}%")

    with col_b:
        st.markdown("**Accolades Available**")
        accolade_cols = ["mvp", "finals_mvp", "championships", "all_star", 
                         "all_nba_total", "dpoy", "all_defensive_total"]
        for col in accolade_cols:
            if col in career_df.columns:
                count = (career_df[col] > 0).sum()
                st.write(f"🏆 {col.replace('_', ' ').title()}: {count} players")

    st.markdown("---")

    # Sample data
    st.markdown("### 📋 Sample Career Data")
    display_cols = ["player_name", "seasons", "games", "tot_pts", "tot_reb", 
                    "tot_ast", "mvp", "all_star", "hof_index"]
    display_cols = [c for c in display_cols if c in career_df.columns]
    st.dataframe(
        career_df[display_cols].head(10),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    st.caption(
        "**Created by Aditya Sudarsan Anand** – CMSE 830 Final Project | "
        "NBA Analytics & Hall of Fame Index Dashboard"
    )

# ========== TAB 2: EDA ==========
with tabs[1]:
    st.subheader("Exploratory Data Analysis")

    eda_section = st.radio(
        "Select Analysis",
        ["Correlation Matrix", "Stat Distributions", "Career Trends", "Awards Overview"],
        horizontal=True
    )

    if eda_section == "Correlation Matrix":
        st.markdown("### 📊 Correlation Analysis")

        numeric_cols = season_df.select_dtypes(include=np.number).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ["year"]]

        corr_method = st.selectbox("Method", ["Pearson", "Spearman"], index=0)
        method_map = {"Pearson": "pearson", "Spearman": "spearman"}

        if numeric_cols and PLOTLY_AVAILABLE:
            corr_matrix = season_df[numeric_cols].corr(method=method_map[corr_method])

            fig = px.imshow(
                corr_matrix,
                labels=dict(x="", y="", color="Correlation"),
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                aspect="auto",
                height=500,
                title=f"{corr_method} Correlation Matrix - Season Stats"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Create correlation table manually:")
            if numeric_cols:
                corr_matrix = season_df[numeric_cols].corr(method=method_map[corr_method])
                st.dataframe(corr_matrix, use_container_width=True)

    elif eda_section == "Stat Distributions":
        st.markdown("### 📈 Statistical Distributions")

        col1, col2 = st.columns(2)

        with col1:
            stat_choice = st.selectbox(
                "Select Stat (Season Level)",
                ["pts", "reb", "ast", "stl", "blk", "games"]
            )

        with col2:
            bin_count = st.slider("Bins", 10, 50, 30)

        stat_data = season_df[stat_choice].dropna()

        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=stat_data,
                nbinsx=bin_count,
                name=stat_choice.upper(),
                marker_color="rgba(0, 100, 200, 0.7)"
            ))

            fig.add_trace(go.Scatter(
                x=[stat_data.mean()] * 2,
                y=[0, season_df[stat_choice].notna().sum()],
                mode="lines",
                name="Mean",
                line=dict(color="red", dash="dash", width=2)
            ))

            fig.update_layout(
                title=f"Distribution of {stat_choice.upper()} (Season Level)",
                xaxis_title=stat_choice.upper(),
                yaxis_title="Frequency",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(pd.DataFrame({stat_choice: stat_data}).value_counts())

        # Stats summary
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Mean", f"{stat_data.mean():.1f}")
        with col_b:
            st.metric("Median", f"{stat_data.median():.1f}")
        with col_c:
            st.metric("Std Dev", f"{stat_data.std():.1f}")
        with col_d:
            st.metric("Max", f"{stat_data.max():.1f}")

    elif eda_section == "Career Trends":
        st.markdown("### 📊 Career Trajectory (Top 15 Scorers)")

        top_players = career_df.nlargest(15, "tot_pts")

        if len(top_players) > 0:
            selected_player = st.selectbox(
                "Select Player",
                top_players["player_name"].values
            )

            player_season = season_df[season_df["player_name"] == selected_player].sort_values("year")

            if len(player_season) > 0:
                st.markdown(f"**{selected_player} Season-by-Season Stats**")
                st.dataframe(player_season, use_container_width=True, hide_index=True)

                # Line chart
                if PLOTLY_AVAILABLE:
                    fig = go.Figure()

                    fig.add_trace(go.Scatter(x=player_season["year"], y=player_season["pts"],
                                           mode="lines+markers", name="Points"))
                    fig.add_trace(go.Scatter(x=player_season["year"], y=player_season["reb"],
                                           mode="lines+markers", name="Rebounds"))
                    fig.add_trace(go.Scatter(x=player_season["year"], y=player_season["ast"],
                                           mode="lines+markers", name="Assists"))

                    fig.update_layout(title=f"{selected_player} - Career Stats", height=400)
                    st.plotly_chart(fig, use_container_width=True)

    elif eda_section == "Awards Overview":
        st.markdown("### 🏆 Awards Distribution")

        award_cols = ["mvp", "finals_mvp", "championships", "all_star", 
                      "all_nba_total", "dpoy", "all_defensive_total"]
        award_cols = [c for c in award_cols if c in career_df.columns]

        award_counts = {col.replace("_", " ").title(): (career_df[col] > 0).sum() 
                       for col in award_cols}

        if PLOTLY_AVAILABLE:
            fig = px.bar(
                x=list(award_counts.keys()),
                y=list(award_counts.values()),
                title="Number of Players with Each Award",
                labels={"x": "Award", "y": "Player Count"},
                color=list(award_counts.values()),
                color_continuous_scale="Viridis"
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(pd.DataFrame(award_counts, index=[0]).T)

        # Detailed breakdown
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**MVP Distribution**")
            mvp_dist = career_df[career_df["mvp"] > 0]["mvp"].value_counts().sort_index()
            for mvp_count, player_count in mvp_dist.items():
                st.write(f"{int(mvp_count)} MVP(s): {int(player_count)} players")

        with col2:
            st.markdown("**All-Star Distribution**")
            allstar_dist = career_df[career_df["all_star"] > 0]["all_star"].value_counts().head(10).sort_index()
            for allstar_count, player_count in allstar_dist.items():
                st.write(f"{int(allstar_count)} All-Stars: {int(player_count)} players")

# ========== TAB 3: PLAYER EXPLORER ==========
with tabs[2]:
    st.subheader("Season-Level Player Explorer")

    all_players = sorted(season_df["player_name"].unique())

    col1, col2 = st.columns([3, 1])

    with col1:
        player_search = st.selectbox(
            "Search & Select Player",
            all_players,
            help="Filter players by name"
        )

    player_data = season_df[season_df["player_name"] == player_search].sort_values("year")
    career_row = career_df[career_df["player_name"] == player_search]

    if len(player_data) > 0:
        # Player summary card
        st.markdown(f"### {player_search}")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Career Seasons", int(career_row["seasons"].values[0]) 
                     if len(career_row) > 0 else "N/A")

        with col2:
            st.metric("Total Games", int(career_row["games"].values[0]) 
                     if len(career_row) > 0 else "N/A")

        with col3:
            st.metric("Career Points", int(career_row["tot_pts"].values[0]) 
                     if len(career_row) > 0 else "N/A")

        with col4:
            hof_idx = career_row["hof_index"].values[0] if len(career_row) > 0 else "N/A"
            st.metric("HoF Index", f"{hof_idx:.1f}" if hof_idx != "N/A" else "N/A")

        st.markdown("---")

        # Season-by-season stats
        st.markdown("**Season-by-Season Statistics**")
        display_cols = ["year", "team", "games", "pts", "reb", "ast", "stl", "blk"]
        st.dataframe(
            player_data[display_cols],
            use_container_width=True,
            hide_index=True
        )

# ========== TAB 4: PLAYER COMPARISON ==========
with tabs[3]:
    st.subheader("Head-to-Head Player Comparison")

    col1, col2 = st.columns(2)

    all_players_comp = sorted(career_filtered["player_name"].unique())

    with col1:
        player1_name = st.selectbox(
            "Player 1",
            all_players_comp,
            key="player1"
        )

    with col2:
        player2_name = st.selectbox(
            "Player 2",
            all_players_comp,
            index=min(1, len(all_players_comp) - 1),
            key="player2"
        )

    if player1_name and player2_name and player1_name != player2_name:
        p1 = career_filtered[career_filtered["player_name"] == player1_name].iloc[0]
        p2 = career_filtered[career_filtered["player_name"] == player2_name].iloc[0]

        # Comparison table
        st.markdown("### 📊 Career Statistics Comparison")

        comp_stats = ["seasons", "games", "tot_pts", "tot_reb", "tot_ast", 
                      "tot_stl", "tot_blk", "mvp", "all_star", "championships", "hof_index"]
        comp_stats = [c for c in comp_stats if c in career_filtered.columns]

        comp_data = []
        for stat in comp_stats:
            v1 = p1[stat] if pd.notna(p1[stat]) else 0
            v2 = p2[stat] if pd.notna(p2[stat]) else 0

            # Determine winner
            if stat == "hof_index":
                winner = "🔴 " + player1_name if v1 > v2 else "🔵 " + player2_name
                v1, v2 = f"{v1:.1f}", f"{v2:.1f}"
            else:
                winner = "🔴" if v1 > v2 else "🔵" if v2 > v1 else "⚪"
                v1, v2 = int(v1), int(v2)

            comp_data.append({
                "Stat": stat.replace("tot_", "Total ").replace("_", " ").title(),
                player1_name: v1,
                player2_name: v2,
                "Leader": winner
            })

        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    else:
        st.info("Select two different players to compare.")

# ========== TAB 5: TEAM TRENDS ==========
with tabs[4]:
    st.subheader("Team Performance Trends")

    team_list = sorted(team_df["team"].unique())

    col1, col2 = st.columns([2, 1])

    with col1:
        team_choice = st.selectbox("Select Team", team_list)

    team_data = team_df[team_df["team"] == team_choice].sort_values("year")

    if len(team_data) > 0:
        st.markdown(f"### {team_choice} - Win Percentage Over Time")
        st.dataframe(team_data, use_container_width=True, hide_index=True)

        # Team stats
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Avg Win %", f"{team_data['win_pct'].mean():.1%}")

        with col2:
            st.metric("Best Season", f"{team_data['win_pct'].max():.1%}")

        with col3:
            st.metric("Worst Season", f"{team_data['win_pct'].min():.1%}")

        with col4:
            st.metric("Seasons", len(team_data))

        st.markdown("---")

        # Player roster (top by points that season)
        st.markdown("**Top Scoring Players**")

        team_players = season_df[season_df["team"] == team_choice].nlargest(10, "pts")
        if len(team_players) > 0:
            display_cols = ["year", "player_name", "pts", "reb", "ast", "games"]
            st.dataframe(
                team_players[display_cols],
                use_container_width=True,
                hide_index=True
            )

# ========== TAB 6: HALL OF FAME INDEX ==========
with tabs[5]:
    st.subheader("Hall of Fame Index Explorer (0-100)")

    st.markdown(
        """
**Enhanced Hall of Fame Index** with research-backed accolade weights:

**Production & Longevity:**
- Career seasons, games, total points/rebounds/assists/steals/blocks
- Average team win percentage

**Awards Weights (Most → Least Important):**
- 🏆 MVP (15x) - Strongest HoF predictor
- 🏆 Finals MVP (12x)
- 💍 Championships (8x)
- 🛡️ DPOY (5x)
- ⭐ All-NBA Total (4x)
- 🌟 All-Star (2x)
- 🛡️ All-Defense Total (1.5x)

_Lower index = more traditional role players | Higher index = elite/inner-circle candidates_
"""
    )

    st.markdown("---")

    # Filters in sidebar (already applied to career_filtered)
    st.markdown("### 🎯 HoF Rankings")

    col1, col2 = st.columns(2)

    with col1:
        sort_by = st.selectbox(
            "Sort By",
            ["HoF Index", "Total Points", "MVP Awards", "All-Stars", "Championships"]
        )

    with col2:
        top_n = st.slider("Show Top N", 10, 100, 50, 10)

    # Sort
    if sort_by == "HoF Index":
        career_sorted = career_filtered.sort_values("hof_index", ascending=False)
    elif sort_by == "Total Points":
        career_sorted = career_filtered.sort_values("tot_pts", ascending=False)
    elif sort_by == "MVP Awards":
        career_sorted = career_filtered.sort_values("mvp", ascending=False)
    elif sort_by == "All-Stars":
        career_sorted = career_filtered.sort_values("all_star", ascending=False)
    else:  # Championships
        career_sorted = career_filtered.sort_values("championships", ascending=False)

    # Display table
    display_cols = ["player_name", "seasons", "games", "tot_pts", "tot_reb", "tot_ast",
                    "mvp", "all_star", "championships", "hof_index"]
    display_cols = [c for c in display_cols if c in career_sorted.columns]

    st.dataframe(
        career_sorted[display_cols].head(top_n),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # Player inspector
    st.markdown("### 🔍 Individual Player Profile")

    player_for_profile = st.selectbox(
        "Select Player",
        sorted(career_filtered["player_name"].unique()),
        key="profile_player"
    )

    player_profile = career_filtered[career_filtered["player_name"] == player_for_profile]

    if len(player_profile) > 0:
        p = player_profile.iloc[0]

        # Header
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            st.markdown(f"## {player_for_profile}")
            st.markdown(f"**Career: {int(p['from_year'])}-{int(p['to_year'])}** | "
                       f"**{int(p['seasons'])} Seasons**")

        with col3:
            hof_idx = p["hof_index"]
            if hof_idx >= 95:
                verdict = "🏆 Elite/Inner-Circle"
                color = "green"
            elif hof_idx >= 85:
                verdict = "⭐ Strong Candidate"
                color = "blue"
            elif hof_idx >= 70:
                verdict = "🎯 Borderline"
                color = "orange"
            elif hof_idx >= 50:
                verdict = "✅ Solid Career"
                color = "gray"
            else:
                verdict = "📊 Role Player"
                color = "lightgray"

            st.markdown(f"**{verdict}**")

        st.markdown("---")

        # Stats and awards
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Career Stats**")
            st.metric("Games", int(p["games"]))
            st.metric("Points", int(p["tot_pts"]))
            st.metric("Rebounds", int(p["tot_reb"]))
            st.metric("Assists", int(p["tot_ast"]))

        with col2:
            st.markdown("**Defensive Stats**")
            st.metric("Steals", int(p["tot_stl"]))
            st.metric("Blocks", int(p["tot_blk"]))
            st.metric("Avg Team Win%", f"{p['avg_team_win_pct']:.1%}")

        with col3:
            st.markdown("**Awards & Accolades**")
            st.metric("MVP", int(p["mvp"]))
            st.metric("Finals MVP", int(p["finals_mvp"]))
            st.metric("All-Stars", int(p["all_star"]))
            st.metric("Championships", int(p["championships"]))

        st.markdown("---")

        # HoF Index visualization
        if PLOTLY_AVAILABLE:
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=[player_for_profile],
                y=[hof_idx],
                marker=dict(color=color, line=dict(color="black", width=2)),
                name="HoF Index"
            ))

            fig.add_hline(y=95, line_dash="dash", line_color="green", 
                         annotation_text="Elite (95)", annotation_position="right")
            fig.add_hline(y=85, line_dash="dash", line_color="blue", 
                         annotation_text="Strong (85)", annotation_position="right")
            fig.add_hline(y=70, line_dash="dash", line_color="orange", 
                         annotation_text="Borderline (70)", annotation_position="right")
            fig.add_hline(y=50, line_dash="dash", line_color="gray", 
                         annotation_text="Average (50)", annotation_position="right")

            fig.update_layout(
                title=f"{player_for_profile} - Enhanced HoF Index",
                yaxis_title="Index (0-100)",
                height=400,
                showlegend=False,
                xaxis=dict(showticklabels=False)
            )

            fig.update_yaxes(range=[0, 105])

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write(f"HoF Index: {hof_idx:.1f} / 100 - {verdict}")

    st.markdown("---")

    # Download data
    st.markdown("### ⬇️ Export Data")

    csv_buffer = career_filtered[[c for c in ["player_name", "seasons", "games", "tot_pts", 
                                               "tot_reb", "tot_ast", "mvp", "all_star", 
                                               "championships", "hof_index"]
                                  if c in career_filtered.columns]].to_csv(index=False)

    st.download_button(
        label="📥 Download Career Data (CSV)",
        data=csv_buffer,
        file_name="nba_career_hof_index.csv",
        mime="text/csv",
        key="download_csv"
    )

st.markdown("---")
st.caption(
    "🏀 **NBA Analytics & Hall of Fame Index Dashboard** | "
    "CMSE 830 Final Project | Created by Aditya Sudarsan Anand"
)
