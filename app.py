    # --- HoF probability heuristic (robust, numeric) ---
    # First build a raw score that is guaranteed to vary whenever the inputs vary.

    # make sure some key columns are numeric
    for col in ["tot_pts", "tot_trb", "tot_ast", "games", "avg_per", "avg_ws", "avg_team_win_pct"]:
        if col in career.columns:
            career[col] = pd.to_numeric(career[col], errors="coerce")

    # z-score helper
    def z(col):
        s = career[col]
        if s.isna().all() or s.std(ddof=0) == 0:
            return pd.Series(0.0, index=career.index)
        return (s - s.mean()) / s.std(ddof=0)

    # start from 0 and add weighted z-scores
    score = pd.Series(0.0, index=career.index)

    if "tot_pts" in career.columns:
        score += 1.0 * z("tot_pts")
    if "tot_trb" in career.columns:
        score += 0.7 * z("tot_trb")
    if "tot_ast" in career.columns:
        score += 0.9 * z("tot_ast")
    if "games" in career.columns:
        score += 0.5 * z("games")
    if "avg_per" in career.columns:
        score += 0.8 * z("avg_per")
    if "avg_ws" in career.columns:
        score += 0.8 * z("avg_ws")
    if "avg_team_win_pct" in career.columns:
        score += 0.6 * z("avg_team_win_pct")

    # if score is still constant (e.g., data very limited), fall back to total points
    if score.nunique() <= 1 and "tot_pts" in career.columns:
        score = z("tot_pts")

    # if *still* constant, just set everything to 0.0
    if score.nunique() <= 1:
        score = pd.Series(0.0, index=career.index)

    career["hof_score_raw"] = score

    # Convert raw score to [0,1] probability-like value using min-max scaling.
    if len(career) > 0:
        s_min = career["hof_score_raw"].min()
        s_max = career["hof_score_raw"].max()
        if s_max == s_min:
            # completely flat: give everyone 0.5
            career["hof_prob"] = 0.5
        else:
            scaled = (career["hof_score_raw"] - s_min) / (s_max - s_min)
            # squeeze slightly away from 0 and 1 so nothing is exactly 0 or 1
            career["hof_prob"] = 0.02 + 0.96 * scaled
    else:
        career["hof_prob"] = []
