# ==================================================
# BASKETBALL ANALYTICS DASHBOARD (NBA 2024–25)
# ==================================================
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# DATA LOAD
# -------------------------
@st.cache_data
def load_data():
    if not os.path.exists("nba_player_stats_2425.csv"):
        raise FileNotFoundError("nba_player_stats_2425.csv not found in current folder.")
    df = pd.read_csv("nba_player_stats_2425.csv")
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
    return df

df = load_data()

# -------------------------
# PAGE CONFIGURATION
# -------------------------
st.set_page_config(
    page_title="Basketball Analytics Dashboard",
    page_icon="🏀",
    layout="wide"
)

st.title("🏀 Basketball Analytics & Player Performance Dashboard")
st.markdown("""
This dashboard explores **NBA player statistics** from the 2024–25 season using data science techniques.
You can:
- Analyze player performance  
- Compare players head-to-head  
- Explore statistical relationships and trends  
""")

# -------------------------
# EDA SECTION
# -------------------------
st.subheader("📊 Exploratory Data Analysis (EDA)")

col1, col2 = st.columns(2)

with col1:
    st.write("### Dataset Overview")
    st.write(df.head())
    st.write("**Shape:**", df.shape)
    st.write("**Missing values per column:**")
    st.dataframe(df.isna().sum())

with col2:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        st.write("### Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns to plot correlation heatmap.")

# -------------------------
# PLAYER COMPARISON
# -------------------------
st.subheader("🆚 Player Comparison")

all_players = sorted(df['player'].dropna().unique())

if len(all_players) >= 2:
    player1 = st.selectbox("Select Player 1", all_players, key="p1")
    player2 = st.selectbox("Select Player 2", all_players, key="p2")

    if player1 and player2 and player1 != player2:
        cols_to_compare = ["pts", "trb", "ast", "stl", "blk", "tov"]
        available_cols = [col for col in cols_to_compare if col in df.columns]

        df_comp = df[df['player'].isin([player1, player2])]
        if not df_comp.empty and available_cols:
            st.write(f"Comparing **{player1}** vs **{player2}**")
            fig, ax = plt.subplots(figsize=(8, 4))
            df_comp_grouped = df_comp.groupby("player")[available_cols].mean().T
            df_comp_grouped.plot(kind="bar", ax=ax)
            plt.title("Average Performance Comparison")
            plt.ylabel("Average Stats")
            plt.xticks(rotation=0)
            st.pyplot(fig)
        else:
            st.warning("Player data not found or no comparable columns available.")

# -------------------------
# PERFORMANCE DISTRIBUTIONS
# -------------------------
st.subheader("📈 Performance Distributions")

if numeric_cols:
    selected_player = st.selectbox("Select Player to Analyze", options=all_players, key="player_dist")
    player_data = df[df['player'] == selected_player]

    if not player_data.empty:
        selected_stat = st.selectbox("Select Stat to Analyze", options=numeric_cols, key="stat_dist")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(player_data[selected_stat].dropna(), kde=True, bins=20, ax=ax)
        plt.title(f"{selected_player} - Distribution of {selected_stat}")
        st.pyplot(fig)
    else:
        st.warning("No data found for the selected player.")
else:
    st.warning("No numeric columns to analyze.")

# -------------------------
# TEAM SUMMARY
# -------------------------
st.subheader("🏟️ Team Summary Statistics")

if 'tm' in df.columns:
    numeric_cols = df.select_dtypes(include=np.number).columns
    team_summary = df.groupby("tm")[numeric_cols].mean().round(2)
    st.dataframe(team_summary.style.background_gradient(cmap="Blues"))

    st.write("### 🧾 All Players (including non-numeric columns)")
    st.dataframe(df.reset_index(drop=True))  # Show all players and all columns
else:
    st.warning("No team column found in dataset.")

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("**Data Source:** [Kaggle - NBA Player Stats 2024–25](https://www.kaggle.com/datasets/eduardopalmieri/nba-player-stats-season-2425)")
st.markdown("Created by: *Aditya Sudarsan Anand* | Data Science Midterm Project")
