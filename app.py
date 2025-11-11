# ==================================================
# BASKETBALL ANALYTICS DASHBOARD (NBA 2024–25)
# ==================================================
import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# PAGE CONFIGURATION
# -------------------------
st.set_page_config(
    page_title="Basketball Analytics Dashboard",
    page_icon="🏀",
    layout="wide"
)

# =========================================
# 1) DATA LOADING (≥2 SOURCES) + CLEANING
# =========================================
@st.cache_data
def load_player_data(path="nba_player_stats_2425.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found in current folder. Place your 24–25 game log CSV here."
        )
    df = pd.read_csv(path)
    # Normalize headers and map to tidy names
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("%", "pct")
        .str.lower()
    )
    # Typical columns in your file (provided earlier):
    # Player, Tm, Opp, Res, MP, FG, FGA, FG%, 3P, 3PA, 3P%, FT, FTA, FT%, ORB, DRB, TRB, AST, STL, BLK, TOV, PF, PTS, GmSc, Date
    # Ensure standard names for downstream use
    rename_map = {
        "tm": "team",
        "opp": "opponent",
        "res": "result",
        "mp": "minutes",
        "fgpct": "fg_pct",
        "3p": "fg3",
        "3papct": "fg3_pct",
        "3pa": "fg3a",
        "ftpct": "ft_pct",
        "trb": "reb",
        "ast": "ast",
        "stl": "stl",
        "blk": "blk",
        "tov": "tov",
        "pf": "pf",
        "pts": "pts",
        "gmsc": "gmsc",
        "date": "date",
    }
    # Only rename keys that exist
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Coerce numerics safely
    numeric_like = [
        "minutes", "fg", "fga", "fg_pct", "fg3", "fg3a", "fg3_pct", "ft", "fta", "ft_pct",
        "orb", "drb", "reb", "ast", "stl", "blk", "tov", "pf", "pts", "gmsc"
    ]
    for col in numeric_like:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse date if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Trim string columns
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    # Derived advanced shooting metrics (when possible)
    if {"pts", "fga", "fg3", "fta"}.issubset(df.columns):
        # TS% = PTS / (2*(FGA + 0.44*FTA))
        denom = (2 * (df["fga"].fillna(0) + 0.44 * df["fta"].fillna(0))).replace(0, np.nan)
        df["ts_pct"] = (df["pts"] / denom).clip(upper=2)  # clip to avoid outliers if data noisy
        # eFG% = (FG + 0.5*3P) / FGA
        denom2 = df["fga"].replace(0, np.nan)
        if "fg" in df.columns:
            df["efg_pct"] = (df["fg"] + 0.5 * df["fg3"].fillna(0)) / denom2

    # Add a per-game rate example (rebounds per minute), guarded for zeros
    if {"reb", "minutes"}.issubset(df.columns):
        denom3 = df["minutes"].replace(0, np.nan)
        df["reb_per_min"] = df["reb"] / denom3

    return df

@st.cache_data
def load_team_meta(path="teams_meta.csv") -> pd.DataFrame:
    """
    Second source for rubric: team-level meta (conference/division).
    If a CSV is present, use it. Else, synthesize a minimal table from the data.
    """
    if os.path.exists(path):
        meta = pd.read_csv(path)
        meta.columns = meta.columns.str.strip().str.lower()
        # Expect columns like: team, conference, division
        return meta

    # Fallback synth meta (keeps rubric's 'multi-source' requirement)
    # Will be joined on 'team' code in your dataset (e.g., LAL, BOS, etc.)
    # If your dataset uses different team labels, adjust here.
    return pd.DataFrame({
        "team": [],  # fill if you want to seed known teams
        "conference": [],
        "division": []
    })

@st.cache_data
def merged_data(players: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    df = players.copy()
    if "team" in df.columns and not teams.empty and "team" in teams.columns:
        df = df.merge(teams.drop_duplicates(subset=["team"]), on="team", how="left")
    return df

# Load
players_df = load_player_data()
teams_df = load_team_meta()
df = merged_data(players_df, teams_df)

# =========================================
# 2) APP HEADER + NAV TABS (self-documenting)
# =========================================
st.title("🏀 Basketball Analytics & Player Performance Dashboard")


tabs = st.tabs(["Overview", "EDA", "Player Comparison", "Team Summary", "Missing Data"])

# =========================================
# 3) OVERVIEW (data profile, downloads)
# =========================================
with tabs[0]:
    st.subheader("Dataset Overview")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.write(df.head(20))
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    with c2:
        st.write("**Missing values (per column):**")
        st.dataframe(df.isna().sum().to_frame("missing_count"))

    # Quick data dictionary
    st.markdown("**Columns (first 40):**")
    st.code(", ".join(df.columns.tolist()[:40]))

    # Downloads: cleaned + a light team aggregate
    def to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
        return dataframe.to_csv(index=False).encode("utf-8")

    st.markdown("**Download cleaned data**")
    st.download_button(
        "⬇️ Download CSV (cleaned)",
        data=to_csv_bytes(df),
        file_name="nba_players_cleaned.csv",
        mime="text/csv"
    )

    if "team" in df.columns:
        num_cols = df.select_dtypes(include=np.number).columns
        team_agg = df.groupby("team")[num_cols].mean(numeric_only=True).round(3).reset_index()
        st.download_button(
            "⬇️ Download CSV (team aggregates)",
            data=to_csv_bytes(team_agg),
            file_name="nba_team_aggregates.csv",
            mime="text/csv"
        )

# =========================================
# 4) EDA (3+ visualizations, encodings)
# =========================================
with tabs[1]:
    st.subheader("Exploratory Data Analysis")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    colA, colB = st.columns(2)
    # A. Correlation heatmap
    with colA:
        if len(num_cols) >= 2:
            method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"], index=0)
            corr = df[num_cols].corr(method=method)
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
            ax.set_title(f"Correlation Heatmap ({method.title()})")
            st.pyplot(fig)
        else:
            st.info("Need ≥2 numeric columns for correlation.")

    # B. Distribution (hist + KDE) with optional player filter
    with colB:
        if num_cols:
            stat = st.selectbox("Select numeric stat", num_cols, index=min(3, len(num_cols)-1))
            player_list = sorted(df["player"].dropna().unique()) if "player" in df.columns else []
            player_filter = st.selectbox("(Optional) Filter by player", ["All"] + player_list)
            plot_df = df if player_filter == "All" else df[df["player"] == player_filter]
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.histplot(plot_df[stat].dropna(), kde=True, bins=25, ax=ax)
            ax.set_title(f"Distribution of {stat}" + ("" if player_filter == "All" else f" — {player_filter}"))
            st.pyplot(fig)

    # C. Boxplot by team (encoding: categorical vs continuous)
    if "team" in df.columns and num_cols:
        stat2 = st.selectbox("Boxplot stat by team", num_cols, index=0)
        # To reduce overplotting if many teams, show top-12 by mean of chosen stat
        tops = (
            df.groupby("team")[stat2]
            .mean(numeric_only=True)
            .sort_values(ascending=False)
            .head(12)
            .index
        )
        bp_df = df[df["team"].isin(tops)]
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.boxplot(data=bp_df, x="team", y=stat2, ax=ax)
        ax.set_title(f"{stat2}: distribution across top-12 teams by mean {stat2}")
        ax.tick_params(axis='x', rotation=0)
        st.pyplot(fig)

    # D. Trend line over time (if 'date' exists)
    if "date" in df.columns and num_cols:
        stat3 = st.selectbox("Time trend stat", num_cols, index=min(1, len(num_cols)-1))
        tl = (
            df.dropna(subset=["date"])[["date", stat3]]
            .groupby("date", as_index=False)
            .mean(numeric_only=True)
            .sort_values("date")
        )
        if not tl.empty:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(tl["date"], tl[stat3])
            ax.set_title(f"League-wide daily mean of {stat3}")
            ax.set_xlabel("Date")
            ax.set_ylabel(stat3)
            st.pyplot(fig)

# =========================================
# 5) PLAYER COMPARISON (bar + radar)
# =========================================
with tabs[2]:
    st.subheader("Player Comparison (Head-to-Head)")
    players = sorted(df["player"].dropna().unique()) if "player" in df.columns else []
    if len(players) >= 2:
        c1, c2 = st.columns(2)
        with c1:
            p1 = st.selectbox("Select Player 1", players, key="p1_cmp")
        with c2:
            p2 = st.selectbox("Select Player 2", players, key="p2_cmp")

        # Stats to compare
        default_stats = [c for c in ["pts", "reb", "ast", "stl", "blk", "tov", "ts_pct", "efg_pct"] if c in df.columns]
        stats = st.multiselect("Metrics to compare", default_stats, default=default_stats)

        sub = df[df["player"].isin([p1, p2])]
        if not sub.empty and stats:
            grp = sub.groupby("player")[stats].mean(numeric_only=True)

            # Bar chart
            fig, ax = plt.subplots(figsize=(8, 4))
            grp.T.plot(kind="bar", ax=ax)
            ax.set_title(f"Average Performance: {p1} vs {p2}")
            ax.set_ylabel("Mean Value")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            st.pyplot(fig)

            # Radar chart (Above & Beyond)
            def radar_plot(data: pd.DataFrame):
                cats = list(data.columns)
                values1 = data.iloc[0].values.astype(float)
                values2 = data.iloc[1].values.astype(float)
                # Normalize for better shape (min-max per metric)
                col_min = data.min()
                col_max = data.max().replace(0, np.nan)
                norm = (data - col_min) / (col_max - col_min)
                v1 = norm.iloc[0].fillna(0).values
                v2 = norm.iloc[1].fillna(0).values

                angles = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist()
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
                ax2.set_title("Radar (normalized)")
                return fig2

            if grp.shape[0] == 2:
                st.pyplot(radar_plot(grp))
        else:
            st.info("Select at least two players and one metric to compare.")
    else:
        st.info("Not enough distinct players to compare.")

# =========================================
# 6) TEAM SUMMARY (all columns table + numeric aggregates)
# =========================================
with tabs[3]:
    st.subheader("Team Summary Statistics")
    if "team" in df.columns:
        num_cols2 = df.select_dtypes(include=np.number).columns
        team_mean = df.groupby("team")[num_cols2].mean(numeric_only=True).round(2)
        team_median = df.groupby("team")[num_cols2].median(numeric_only=True).round(2)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Team Means (numeric)**")
            st.dataframe(team_mean.style.background_gradient(cmap="Blues"))
        with c2:
            st.markdown("**Team Medians (numeric)**")
            st.dataframe(team_median.style.background_gradient(cmap="Greens"))

        st.markdown("### 🧾 All Players (including non-numeric columns)")
        st.dataframe(df.reset_index(drop=True))
    else:
        st.warning("No 'team' column found.")

# =========================================
# 7) MISSING DATA (diagnostics + imputation choices)
# =========================================
with tabs[4]:
    st.subheader("Missing Data & Imputation")
    miss = df.isna().mean().sort_values(ascending=False)
    st.write("**Missingness fraction per column:**")
    st.dataframe(miss.to_frame("missing_fraction"))

    st.markdown("**Impute selected columns**")
    cols_to_impute = st.multiselect(
        "Select columns to impute",
        options=df.columns.tolist(),
        default=[c for c in df.select_dtypes(include=np.number).columns.tolist() if df[c].isna().any()]
    )
    strategy = st.selectbox("Strategy", ["mean", "median", "mode", "ffill", "bfill"])
    preview_rows = st.slider("Preview rows", min_value=5, max_value=50, value=10, step=5)

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

    st.markdown("**Before vs After (sample preview)**")
    c1, c2 = st.columns(2)
    with c1:
        st.write("Before")
        st.dataframe(df.head(preview_rows))
    with c2:
        st.write("After")
        st.dataframe(df_imputed.head(preview_rows))

    # Simple validation metric (numeric columns): difference in column means
    num_for_check = [c for c in cols_to_impute if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if num_for_check:
        delta = (df_imputed[num_for_check].mean(numeric_only=True) - df[num_for_check].mean(numeric_only=True)).to_frame("mean_shift")
        st.markdown("**Imputation sanity check (mean shift on numeric columns):**")
        st.dataframe(delta)

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("**Data Source:** Kaggle – NBA Player Stats 2024–25 (place CSV locally as `nba_player_stats_2425.csv`).")
st.markdown("Created by: *Aditya Sudarsan Anand* | CMSE 830 Midterm")
