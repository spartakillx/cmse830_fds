# ==================================================
# CMSE 830 FINAL PROJECT - SIMPLIFIED & CRASH-PROOF
# NBA MULTI-DATASET ANALYTICS + HOF INDEX DASHBOARD
# ==================================================
import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="NBA Analytics & Hall of Fame Index",
    page_icon="🏀",
    layout="wide"
)

st.title("🏀 NBA Analytics & Hall of Fame Index Dashboard")

# Add error handling wrapper
try:
    import kagglehub
    import matplotlib.pyplot as plt
    import seaborn as sns
    IMPORTS_OK = True
except Exception as e:
    st.error(f"Import error: {e}")
    IMPORTS_OK = False
    st.stop()

st.markdown(
    """
This app combines **multiple NBA datasets** including **awards and accolades**.

_Note: If you see errors, make sure Kaggle API credentials are set up._
"""
)

# --------------------------------------------------
# HELPER FUNCTIONS WITH ROBUST ERROR HANDLING
# --------------------------------------------------
def load_kaggle_csv(dataset_id: str, prefer_contains=None) -> pd.DataFrame:
    """Download a Kaggle dataset and return the first (or preferred) CSV."""
    try:
        path = kagglehub.dataset_download(dataset_id)
        files = [f for f in os.listdir(path) if f.lower().endswith(".csv")]
        if not files:
            return pd.DataFrame()
        
        csv_name = files[0]
        if prefer_contains:
            for f in files:
                if prefer_contains.lower() in f.lower():
                    csv_name = f
                    break
        
        csv_path = os.path.join(path, csv_name)
        return pd.read_csv(csv_path)
    except Exception as e:
        st.warning(f"Failed to load {dataset_id}: {str(e)[:100]}")
        return pd.DataFrame()


@st.cache_data(show_spinner=True)
def load_all_raw():
    """Load datasets with full error handling."""
    st.info("Loading datasets from Kaggle...")
    
    players_raw = load_kaggle_csv("drgilermo/nba-players-stats")
    if not players_raw.empty:
        st.success(f"✅ Loaded players: {len(players_raw)} rows")
    
    boxscores_raw = load_kaggle_csv("szymonjwiak/nba-traditional")
    if not boxscores_raw.empty:
        st.success(f"✅ Loaded boxscores: {len(boxscores_raw)} rows")
    
    seasons_raw = load_kaggle_csv("boonpalipatana/nba-season-records-from-every-year")
    if not seasons_raw.empty:
        st.success(f"✅ Loaded team records: {len(seasons_raw)} rows")
    
    accolades_raw = load_kaggle_csv("ryanschubertds/all-nba-aba-players-bio-stats-accolades")
    if not accolades_raw.empty:
        st.success(f"✅ Loaded accolades: {len(accolades_raw)} rows")
        st.info(f"Accolade columns: {', '.join(accolades_raw.columns.tolist()[:15])}")
    
    mvp_raw = load_kaggle_csv("robertsunderhaft/nba-player-season-statistics-with-mvp-win-share")
    allstar_raw = load_kaggle_csv("ahmedbendaly/nba-all-star-game-data", prefer_contains="players")

    return players_raw, boxscores_raw, seasons_raw, accolades_raw, mvp_raw, allstar_raw


# Load data with comprehensive error handling
try:
    players_raw, boxscores_raw, seasons_raw, accolades_raw, mvp_raw, allstar_raw = load_all_raw()
except Exception as e:
    st.error(f"Critical error loading data: {e}")
    st.stop()

# Check if we have minimum required data
if players_raw.empty:
    st.error("❌ Failed to load player data. Please check Kaggle API setup.")
    st.info("Make sure you have kagglehub installed and Kaggle API credentials configured.")
    st.stop()

st.success(f"✅ Data loaded successfully! {len(players_raw)} players found.")

# --------------------------------------------------
# BUILD SIMPLE CAREER TABLE
# --------------------------------------------------
try:
    career_df = players_raw.copy()
    career_df.columns = career_df.columns.str.strip().str.lower()
    
    # Standardize names
    if "player" in career_df.columns and "player_name" not in career_df.columns:
        career_df = career_df.rename(columns={"player": "player_name"})
    if "from" in career_df.columns:
        career_df = career_df.rename(columns={"from": "from_year"})
    if "to" in career_df.columns:
        career_df = career_df.rename(columns={"to": "to_year"})
    if "yrs" in career_df.columns and "seasons" not in career_df.columns:
        career_df = career_df.rename(columns={"yrs": "seasons"})
    if "g" in career_df.columns and "games" not in career_df.columns:
        career_df = career_df.rename(columns={"g": "games"})
    
    # Convert to numeric
    for col in ["from_year", "to_year", "seasons", "games", "tot_pts", "tot_trb", "tot_ast"]:
        if col in career_df.columns:
            career_df[col] = pd.to_numeric(career_df[col], errors="coerce")
    
    st.success(f"✅ Processed career data: {len(career_df)} players")
    
except Exception as e:
    st.error(f"Error processing career data: {e}")
    st.stop()

# --------------------------------------------------
# PROCESS ACCOLADES IF AVAILABLE
# --------------------------------------------------
if not accolades_raw.empty:
    try:
        st.info("Processing accolades...")
        accolades = accolades_raw.copy()
        accolades.columns = accolades.columns.str.strip().str.lower()
        
        # Show what columns we have
        st.write("**Accolade columns found:**", list(accolades.columns))
        
        # Standardize player name
        if "player" in accolades.columns and "player_name" not in accolades.columns:
            accolades = accolades.rename(columns={"player": "player_name"})
        
        if "player_name" in accolades.columns:
            # Find relevant columns
            accolade_cols = {}
            
            # Look for key accolades
            for col in accolades.columns:
                col_lower = col.lower()
                if "championship" in col_lower or "ring" in col_lower:
                    accolade_cols["championships"] = col
                elif "mvp" in col_lower and "finals" not in col_lower:
                    accolade_cols["mvp"] = col
                elif "finals" in col_lower and "mvp" in col_lower:
                    accolade_cols["finals_mvp"] = col
                elif "all-star count" in col_lower or "all_star_count" in col_lower:
                    accolade_cols["all_star"] = col
                elif "dpoy" in col_lower or "defensive player" in col_lower:
                    accolade_cols["dpoy"] = col
            
            st.write("**Mapped accolades:**", accolade_cols)
            
            # Merge accolades if we found any
            if accolade_cols:
                # Select and rename columns
                merge_cols = ["player_name"] + list(accolade_cols.values())
                accolades_subset = accolades[merge_cols].copy()
                
                # Rename to standard names
                rename_dict = {v: k for k, v in accolade_cols.items()}
                accolades_subset = accolades_subset.rename(columns=rename_dict)
                
                # Convert to numeric and aggregate
                for col in accolade_cols.keys():
                    if col in accolades_subset.columns:
                        accolades_subset[col] = pd.to_numeric(accolades_subset[col], errors="coerce").fillna(0)
                
                # Aggregate by player
                accolades_agg = accolades_subset.groupby("player_name", as_index=False).sum()
                
                # Merge with career
                career_df = career_df.merge(accolades_agg, on="player_name", how="left")
                
                # Fill NaN with 0
                for col in accolade_cols.keys():
                    if col in career_df.columns:
                        career_df[col] = career_df[col].fillna(0)
                
                st.success(f"✅ Merged accolades for {len(accolades_agg)} players")
            else:
                st.warning("Could not find standard accolade columns")
        
    except Exception as e:
        st.error(f"Error processing accolades: {e}")
        st.exception(e)

# --------------------------------------------------
# CALCULATE HOF INDEX
# --------------------------------------------------
try:
    base_features = [c for c in ["seasons", "games", "tot_pts", "tot_trb", "tot_ast"] 
                     if c in career_df.columns]
    accolade_features = [c for c in ["mvp", "finals_mvp", "championships", "all_star", "dpoy"]
                         if c in career_df.columns]
    
    all_features = base_features + accolade_features
    
    if all_features:
        X = career_df[all_features].fillna(0).astype(float)
        
        # Apply weights to accolades
        X_weighted = X.copy()
        weights = {"mvp": 15, "finals_mvp": 12, "championships": 8, "all_star": 2, "dpoy": 5}
        for accolade, weight in weights.items():
            if accolade in X_weighted.columns:
                X_weighted[accolade] = X_weighted[accolade] * weight
        
        # Z-score
        mean = X_weighted.mean()
        std = X_weighted.std(ddof=0) + 1e-10
        z = (X_weighted - mean) / std
        z = z.fillna(0.0)
        
        raw_score = z.sum(axis=1)
        ranks = raw_score.rank(method="average", pct=True)
        career_df["hof_index"] = (ranks * 100).round(1)
        
        st.success(f"✅ Calculated HoF Index using: {', '.join(all_features)}")
    else:
        career_df["hof_index"] = 0
        st.warning("Not enough features for HoF Index")
        
except Exception as e:
    st.error(f"Error calculating HoF Index: {e}")
    career_df["hof_index"] = 0

# --------------------------------------------------
# DISPLAY
# --------------------------------------------------
st.markdown("---")
st.subheader("Hall of Fame Explorer")

st.write(f"**Total players:** {len(career_df):,}")

if "player_name" in career_df.columns:
    all_players = sorted(career_df["player_name"].dropna().unique())
    
    if all_players:
        selected = st.selectbox(f"Select from {len(all_players):,} players", all_players)
        
        player_row = career_df[career_df["player_name"] == selected]
        
        if not player_row.empty:
            p = player_row.iloc[0]
            
            st.markdown(f"## {selected}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Career Overview**")
                for stat in ["from_year", "to_year", "seasons", "games"]:
                    if stat in p.index and pd.notna(p[stat]):
                        st.write(f"{stat.replace('_', ' ').title()}: {int(p[stat])}")
            
            with col2:
                st.markdown("**Career Totals**")
                for stat in ["tot_pts", "tot_trb", "tot_ast"]:
                    if stat in p.index and pd.notna(p[stat]):
                        st.write(f"{stat.replace('tot_', '').upper()}: {int(p[stat]):,}")
            
            with col3:
                st.markdown("**Accolades** 🏆")
                for accolade in ["mvp", "finals_mvp", "championships", "all_star", "dpoy"]:
                    if accolade in p.index and pd.notna(p[accolade]) and p[accolade] > 0:
                        st.write(f"{accolade.replace('_', ' ').title()}: {int(p[accolade])}")
            
            if "hof_index" in p.index:
                hof_val = float(p["hof_index"])
                st.markdown(f"### HoF Index: **{hof_val:.1f} / 100**")
                
                if hof_val >= 95:
                    st.success("🏆 Elite / Inner-circle HoF")
                elif hof_val >= 85:
                    st.info("⭐ Strong HoF candidate")
                elif hof_val >= 70:
                    st.warning("🎯 Borderline HoF")
                elif hof_val >= 50:
                    st.info("✅ Solid career")
                else:
                    st.info("📊 Role player")

# Top players table
st.markdown("---")
st.subheader("Top 50 Players by HoF Index")

if "hof_index" in career_df.columns:
    top_df = career_df.sort_values("hof_index", ascending=False).head(50)
    
    display_cols = [c for c in ["player_name", "from_year", "to_year", "seasons", "games",
                                 "tot_pts", "tot_trb", "tot_ast", "mvp", "championships", 
                                 "all_star", "hof_index"] if c in top_df.columns]
    
    st.dataframe(top_df[display_cols], use_container_width=True)

st.markdown("---")
st.markdown("Created by **Aditya Sudarsan Anand** – CMSE 830 Final Project")
