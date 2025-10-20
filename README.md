# 🏀 NBA Basketball Analytics Dashboard (2024–25)

## Overview
This dashboard explores NBA player statistics from the 2024–25 season. It allows users to:

- Analyze individual player performance  
- Compare players head-to-head  
- Explore statistical relationships and trends across the league  

Built with Python, Streamlit, Pandas, Seaborn, and Matplotlib.

---

## Features
- Exploratory Data Analysis (EDA): View dataset overview, missing values, and correlation heatmaps  
- Player Comparison: Compare average stats of two players side-by-side  
- Performance Distributions: Analyze distributions of stats for individual players  
- Team Summary Statistics: View numeric averages and all columns per team  

---

## Project Structure

nba-dashboard/
│
├─ app.py                     # Main Streamlit application
├─ nba_player_stats_2425.csv  # Player statistics CSV dataset
├─ README.md                  # Project documentation
├─ requirements.txt           # Python dependencies
└─ images/                    # Optional folder for screenshots or plots

---

## How to Run

1. Install dependencies:
pip install streamlit pandas matplotlib seaborn

2. Navigate to the project folder:
cd path/to/nba-dashboard

3. Run the Streamlit app:
streamlit run app.py

---

## Data Source

Dataset: Kaggle - NBA Player Stats 2024–25
https://www.kaggle.com/datasets/eduardopalmieri/nba-player-stats-season-2425

---

## Features

- Exploratory Data Analysis (EDA): View dataset overview, missing values, and correlation heatmaps  
- Player Comparison: Compare average stats of two players side-by-side  
- Performance Distributions: Analyze distributions of stats for individual players  
- Team Summary Statistics: View numeric averages and all columns per team  

---

## Author

Aditya Sudarsan Anand  
Data Science Midterm Project

---

## Future Improvements

- Add interactive team vs team comparisons  
- Include advanced metrics like Player Efficiency Rating (PER)  
- Improve visualization aesthetics using Plotly for interactivity  
- Add season trends and filtering by game dates  

---

## Notes

- All numeric and non-numeric columns are available in the team summary section.  
- The app allows head-to-head player comparisons and individual performance distribution analysis.  
- Designed for easy exploration of the 2024–25 NBA season dataset.
