# ==================================================
# NBA FINAL PROJECT DASHBOARD (2004–05+)
# Uses 3 Kaggle datasets + HoF probability score
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
    If prefer_contains is provided, prefer CSV whose filename contains that substring.
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
    Keep rows with season_start >= SEASON_START_YEAR when that column is usable.
    If not usable, return df unchanged.
    """
    if "season_start" not in df.columns:
        return df
    if not df["season_start"].notna().any():
        return df
    return df[df["season_start"] >= SEASON_START_YEAR].copy()


# ==================================================
# 2. LOAD THREE DATASETS
# ==================================================
@st.cache_data
def load_players_season_stats() -> pd.DataFrame:
    """
    Dataset 1: drgilermo/nba-players-stats
    Season-by-season player stats.
    """
    df = download_first_csv("drgilermo/nba-players-stats", prefer_contains="Seasons")
    df = normalize_columns(df)

    # Standardize player/team names
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
    """
    Dataset 2: szymonjwiak/nba-traditional
    Game/season level 'traditional' stats.
    """
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
    """
    Dataset 3: boonpalipatana/nba-season-records-from-every-year
    Team season wins/losses.
    """
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


# ==================================================
# 3. BUILD MERGED & CAREER TABLES + HOF PROB
# ==================================================
@st.cache_data
def build_merged_data():
    players = load_players_season_stats()
    trad = load_traditional_box()
    teams = load_team_records()

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
    else:
        career = pd.DataFrame(columns=["player_name"])

    # --- HoF probability heuristic ---
    def safe_log(x):
        return np.log1p(np.maximum(x, 0))

    pieces = []
    if "tot_pts" in career.columns:
        pieces.append(1.0 * safe_log(career["tot_pts"] / 1000))
    if "tot_trb" in career.columns:
        pieces.append(0.7 * safe_log(career["tot_trb"] / 500))
    if "tot_ast" in career.columns:
        pieces.append(0.9 * safe_log(career["tot_ast"] / 500))
    if "games" in career.columns:
        pieces.append(0.4 * safe_log(career["games"] / 200))
    if "avg_per" in career.columns:
        pieces.append(0.8 * (career["avg_per"] - 15) / 10)
    if "avg_ws" in career.columns:
        pieces.append(0.8 * career["avg_ws"] / 5)
    if "avg_team_win_pct" in career.columns:
        pieces.append(0.6 * (career["avg_team_win_pct"] - 0.5) * 2)

    if pieces:
        raw_score = np.sum(pieces, axis=0)
    else:
        raw_score = pd.Series(0, index=career.index)

    career["hof_score_raw"] = raw_score
    if len(career) > 0:
        career["hof_prob"] = career["hof_score_raw"].rank(pct=True)
    else:
        career["hof_prob"] = []

    return merged, trad, teams, career


merged_df, trad_df, teams_df, career_df = build_merged_data()

# ==================================================
# 4. APP TABS
# ==================================================
st.title("🏀 NBA Analytics & Hall of Fame Explorer (2004–05 and later)")
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
        st.markdown("**Season-level player stats (merged with team win% where available):**")
        st.dataframe(merged_df.head(30))
        st.write(f"Rows: {merged_df.shape[0]:,}  |  Columns: {merged_df.shape[1]}")

    with c2:
        st.markdown("**Career-level table (aggregated):**")
        st.dataframe(career_df.head(20))
        st.write(f"Players: {career_df.shape[0]:,}")

    st.markdown("**Columns (first 40):**")
    st.code(", ".join(merged_df.columns.tolist()[:40]))

    st.markdown("---")
    st.markdown(
        "Data sources: Kaggle – "
        "`drgilermo/nba-players-stats`, `szymonjwiak/nba-traditional`, "
        "`boonpalipatana/nba-season-records-from-every-year`."
    )
    st.markdown("Created by **Aditya Sudarsan Anand** – CMSE 830 Final Project.")


# --------------------------------------------------
# TAB 2 – EDA (Players & Teams)
# --------------------------------------------------
with tabs[1]:
    st.subheader("Exploratory Data Analysis")

    num_cols = merged_df.select_dtypes(include=np.number).columns.tolist()

    colA, colB = st.columns(2)

    # A. correlation
    with colA:
        st.markdown("**Correlation heatmap (players, numeric stats)**")
        if len(num_cols) >= 2:
            method = st.selectbox("Correlation method", ["pearson", "spearman"], index=0)
            corr = merged_df[num_cols].corr(method=method)
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
            ax.set_title(f"Correlation ({method})")
            st.pyplot(fig)
        else:
            st.info("Need at least two numeric columns for correlation.")

    # B. Stat distribution
    with colB:
        st.markdown("**Stat distribution (histogram + KDE)**")
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
    st.subheader("Team-level win% distribution (from team records)")
    if "win_pct" in teams_df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(teams_df["win_pct"].dropna(), kde=True, bins=25, ax=ax)
        ax.set_title("Distribution of team win percentage")
        st.pyplot(fig)
    else:
        st.info("Team win_pct column not found in records dataset.")


# --------------------------------------------------
# TAB 3 – PLAYER COMPARISON
# --------------------------------------------------
with tabs[2]:
    st.subheader("Player comparison (head-to-head)")

    if "player_name" not in career_df.columns or career_df.empty:
        st.info("Career table is empty or missing player_name.")
    else:
        players = sorted(career_df["player_name"].dropna().unique())  # full list
        if len(players) < 2:
            st.info("Not enough players in career table.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                p1 = st.selectbox("Player 1", players, key="cmp_p1")
            with c2:
                p2 = st.selectbox("Player 2", players, key="cmp_p2")

            stats_candidates = [
                c for c in [
                    "tot_pts", "tot_trb", "tot_ast", "tot_stl", "tot_blk",
                    "tot_tov", "games", "hof_prob"
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

                # bar chart
                fig, ax = plt.subplots(figsize=(8, 4))
                sub.T.plot(kind="bar", ax=ax)
                ax.set_title(f"{p1} vs {p2}")
                ax.set_ylabel("Value")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                st.pyplot(fig)

                # radar chart (normalized)
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
        st.info("Team records not available in expected format.")
    else:
        teams = sorted(teams_df["team"].dropna().unique())
        selected_team = st.selectbox("Select team", teams)
        sub = teams_df[teams_df["team"] == selected_team].sort_values("season_start")

        if "win_pct" in sub.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(sub["season_start"], sub["win_pct"], marker="o")
            ax.set_xlabel("Season start year")
            ax.set_ylabel("Win%")
            ax.set_title(f"{selected_team} win% over time (>= {SEASON_START_YEAR})")
            st.pyplot(fig)
        else:
            st.info("win_pct column missing in team dataset.")


# --------------------------------------------------
# TAB 5 – HALL OF FAME EXPLORER
# --------------------------------------------------
with tabs[4]:
    st.subheader("Hall of Fame probability explorer")

    st.markdown(
        "This is **not an official HoF model** – it's a heuristic score based on career totals, "
        "efficiency (if available), longevity, and team success, scaled into a 0–1 probability-like value."
    )

    if career_df.empty:
        st.info("Career table is empty – check that player-season data loaded correctly.")
    else:
        max_slider = int(max(10, len(career_df)))  # allow up to all players
        top_n = st.slider(
            "Show top N players",
            min_value=10,
            max_value=max_slider,
            value=min(50, max_slider),
            step=10
        )

        st.markdown(f"**Top {top_n} players by HoF probability**")
        st.dataframe(
            career_df.sort_values("hof_prob", ascending=False).head(top_n)
        )

        # full list of players in dropdown
        players = sorted(career_df["player_name"].dropna().unique())
        sel = st.selectbox("Inspect a player", players, key="hof_player")
        row = career_df[career_df["player_name"] == sel]

        if not row.empty:
            r = row.iloc[0]

            st.markdown(f"## {sel}")

            seasons_val = r.get("seasons", np.nan)
            games_val = r.get("games", np.nan)
            if pd.notna(seasons_val):
                st.write(f"Seasons: {int(seasons_val)}")
            if pd.notna(games_val):
                st.write(f"Games: {int(games_val):,}")
            if "tot_pts" in r and pd.notna(r["tot_pts"]):
                st.write(f"Total points: {int(r['tot_pts']):,}")
            if "tot_trb" in r and pd.notna(r["tot_trb"]):
                st.write(f"Total rebounds: {int(r['tot_trb']):,}")
            if "tot_ast" in r and pd.notna(r["tot_ast"]):
                st.write(f"Total assists: {int(r['tot_ast']):,}")
            if "avg_team_win_pct" in r and pd.notna(r["avg_team_win_pct"]):
                st.write(f"Average team win%: {r['avg_team_win_pct']:.3f}")

            # --- CLEAR HoF probability display ---
            hof_prob = float(r.get("hof_prob", np.nan))
            if not np.isnan(hof_prob):
                pct = hof_prob * 100
                st.markdown(f"### Estimated HoF probability (relative to dataset): **{pct:.1f}%**")

                fig, ax = plt.subplots(figsize=(6, 1.2))
                ax.barh([0], [hof_prob], height=0.4)
                ax.set_xlim(0, 1)
                ax.set_yticks([])
                ax.set_xlabel("HoF probability (0–1)")
                st.pyplot(fig)


# --------------------------------------------------
# TAB 6 – MISSING DATA & IMPUTATION
# --------------------------------------------------
with tabs[5]:
    st.subheader("Missing data & imputation (player-season table)")

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
        st.markdown("**Before imputation**")
        st.dataframe(merged_df.head(n_preview))
    with c2:
        st.markdown("**After imputation**")
        st.dataframe(df_imp.head(n_preview))

    num_cols_imp = [c for c in cols_to_impute if pd.api.types.is_numeric_dtype(merged_df[c])]
    if num_cols_imp:
        delta = (df_imp[num_cols_imp].mean() - merged_df[num_cols_imp].mean()).to_frame("mean_shift")
        st.markdown("**Sanity check: mean shift for numeric columns**")
        st.dataframe(delta)


# --------------------------------------------------
# TAB 7 – DOWNLOADS
# --------------------------------------------------
with tabs[6]:
    st.subheader("Download cleaned tables")

    def to_csv_bytes(df_in: pd.DataFrame) -> bytes:
        return df_in.to_csv(index=False).encode("utf-8")

    st.markdown("**Download player-season merged table**")
    st.download_button(
        label="⬇️ Player-season table (merged)",
        data=to_csv_bytes(merged_df),
        file_name="player_season_merged_2004plus.csv",
        mime="text/csv"
    )

    st.markdown("**Download career-level table with HoF probabilities**")
    career_download = career_df.copy()
    preferred_cols = [
        "player_name", "from_year", "to_year", "seasons", "games",
        "tot_pts", "tot_trb", "tot_ast", "hof_prob"
    ]
    ordered = [c for c in preferred_cols if c in career_download.columns]
    remaining = [c for c in career_download.columns if c not in ordered]
    career_download = career_download[ordered + remaining]

    st.download_button(
        label="⬇️ Career table with HoF probabilities",
        data=to_csv_bytes(career_download),
        file_name="career_with_hof_probabilities.csv",
        mime="text/csv"
    )

    st.markdown("**Download team records table (win%)**")
    st.download_button(
        label="⬇️ Team records",
        data=to_csv_bytes(teams_df),
        file_name="team_records_2004plus.csv",
        mime="text/csv"
    )
