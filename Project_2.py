# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 22:10:39 2026

@author: senaa
"""

# ======================================================
# IMPORTS
# ======================================================
import os
import json
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

# ======================================================
# GLOBAL CONSTANTS
# ======================================================
EVENTS_DIR = "event_data"
SHOT_WINDOW = 15
MAX_SHOTS_PER_PASS = 4

X_GOAL = 100
Y_CENTER = 50
Y_LEFT, Y_RIGHT = 37, 63

DANGER_THRESHOLD = 0.02

# ======================================================
# 1️⃣ PASS → FUTURE SHOT LINKING
# ======================================================
def build_pass_shot_dataset():
    print("1️⃣ Building pass → future shot dataset")

    minutes_df = pd.read_parquet("minutes.parquet")
    minutes_df = minutes_df[["player_id", "match_id", "minutes_played"]]

    all_passes = []

    for filename in os.listdir(EVENTS_DIR):
        if not filename.endswith(".json"):
            continue

        with open(os.path.join(EVENTS_DIR, filename), encoding="utf-8") as f:
            data = json.load(f)

        events = data.get("events", [])
        if not events:
            continue

        match_id = events[0].get("matchId")

        passes, shots = [], []

        for event in events:
            t = event.get("minute", 0) * 60 + event.get("second", 0)

            if event.get("shot") is not None:
                shots.append({
                    "shotEventId": event.get("id"),
                    "shotTeamId": event.get("team", {}).get("id"),
                    "shotTimeSeconds": t,
                    "shotX": event.get("location", {}).get("x"),
                    "shotY": event.get("location", {}).get("y"),
                    "shotIsGoal": event.get("shot", {}).get("isGoal"),
                    "shotXG": event.get("shot", {}).get("xg"),
                })

            if event.get("pass") is not None:
                passes.append({
                    "matchId": match_id,
                    "passEventId": event.get("id"),
                    "passTeamId": event.get("team", {}).get("id"),
                    "passPlayerId": event.get("player", {}).get("id"),
                    "passPlayerName": event.get("player", {}).get("name"),
                    "passPlayerPosition": event.get("player", {}).get("position"),
                    "passTimeSeconds": t,
                    "startX": event.get("location", {}).get("x"),
                    "startY": event.get("location", {}).get("y"),
                    "endX": event.get("pass", {}).get("endLocation", {}).get("x"),
                    "endY": event.get("pass", {}).get("endLocation", {}).get("y"),
                    "becameShot": False,
                    **{f"{k}_{i+1}": None
                       for i in range(MAX_SHOTS_PER_PASS)
                       for k in ["shotId", "shotX", "shotY", "shotIsGoal", "shotXG"]}
                })

        if not passes:
            continue

        passes_df = pd.DataFrame(passes)
        shots_df = pd.DataFrame(shots)

        for idx, p in passes_df.iterrows():
            future = shots_df[
                (shots_df["shotTeamId"] == p["passTeamId"]) &
                (shots_df["shotTimeSeconds"] > p["passTimeSeconds"]) &
                (shots_df["shotTimeSeconds"] <= p["passTimeSeconds"] + SHOT_WINDOW)
            ].sort_values("shotTimeSeconds")

            if future.empty:
                continue

            passes_df.at[idx, "becameShot"] = True

            for i, (_, s) in enumerate(future.head(MAX_SHOTS_PER_PASS).iterrows()):
                j = i + 1
                for col in ["shotEventId", "shotX", "shotY", "shotIsGoal", "shotXG"]:
                    passes_df.at[idx, f"{col.replace('EventId','Id')}_{j}"] = s[col]

        all_passes.extend(passes_df.to_dict("records"))

    df = pd.DataFrame(all_passes)

    minutes_unique = minutes_df.drop_duplicates(["player_id", "match_id"])
    df = df.merge(
        minutes_unique,
        left_on=["passPlayerId", "matchId"],
        right_on=["player_id", "match_id"],
        how="left"
    ).rename(columns={"minutes_played": "minutesPlayed"}).drop(columns=["player_id", "match_id"])

    df.to_csv("all_passes_with_future_shots_detailed.csv", index=False)
    print(" Saved all_passes_with_future_shots_detailed.csv")

# ======================================================
# 2️⃣ LOGISTIC REGRESSION: PASS → SHOT
# ======================================================
def train_pass_to_shot_model():
    print("2️⃣ Training pass → shot model")

    df = pd.read_csv("all_passes_with_future_shots_detailed.csv")

    df["dist_to_goal_start"] = np.sqrt((X_GOAL - df["startX"])**2 + (Y_CENTER - df["startY"])**2)
    df["dist_to_goal_end"] = np.sqrt((X_GOAL - df["endX"])**2 + (Y_CENTER - df["endY"])**2)
    df["log_dist_to_goal_start"] = np.log1p(df["dist_to_goal_start"])
    df["log_dist_to_goal_end"] = np.log1p(df["dist_to_goal_end"])

    df["goal_angle_start"] = np.abs(
        np.arctan2(Y_LEFT - df["startY"], X_GOAL - df["startX"]) -
        np.arctan2(Y_RIGHT - df["startY"], X_GOAL - df["startX"])
    )
    df["goal_angle_end"] = np.abs(
        np.arctan2(Y_LEFT - df["endY"], X_GOAL - df["endX"]) -
        np.arctan2(Y_RIGHT - df["endY"], X_GOAL - df["endX"])
    )

    df["goal_weighted_centrality_start"] = (1 - np.abs(df["startY"] - Y_CENTER)/Y_CENTER) * (df["startX"]/X_GOAL)
    df["goal_weighted_centrality_end"] = (1 - np.abs(df["endY"] - Y_CENTER)/Y_CENTER) * (df["endX"]/X_GOAL)
    df["pass_direction"] = np.arctan2(df["endY"] - df["startY"], df["endX"] - df["startX"])

    features = [
        "log_dist_to_goal_start", "goal_angle_start", "goal_weighted_centrality_start",
        "log_dist_to_goal_end", "goal_angle_end", "goal_weighted_centrality_end",
        "pass_direction"
    ]

    X = df[features]
    y = df["becameShot"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(C=0.5, max_iter=2000)
    model.fit(X_train_s, y_train)

    probs = model.predict_proba(X_test_s)[:, 1]
    print("ROC-AUC:", roc_auc_score(y_test, probs))

    joblib.dump(model, "logreg_pass_to_shot_model.joblib")
    joblib.dump(scaler, "logreg_pass_to_shot_scaler.joblib")

# ======================================================
# 3️⃣ LINEAR REGRESSION: SHOT → GOAL
# ======================================================
def train_shot_to_goal_model():
    print("3️⃣ Training shot → goal model")

    df = pd.read_csv("all_passes_with_future_shots_detailed.csv")
    df = df[df["becameShot"] == True]

    shots = []
    for i in range(1, MAX_SHOTS_PER_PASS + 1):
        tmp = df[df[f"shotX_{i}"].notnull()][
            ["matchId", "passEventId", "startX", "startY", f"shotX_{i}", f"shotY_{i}", f"shotXG_{i}"]
        ].rename(columns={
            "startX": "shotStartX",
            "startY": "shotStartY",
            f"shotX_{i}": "shotX",
            f"shotY_{i}": "shotY",
            f"shotXG_{i}": "shotXG"
        })
        shots.append(tmp)

    shots_df = pd.concat(shots)
    shots_df = shots_df[shots_df["shotXG"].notnull()]

    dx = X_GOAL - shots_df["shotX"]
    dy = shots_df["shotY"] - Y_CENTER

    shots_df["dist_to_goal"] = np.sqrt(dx**2 + dy**2)
    shots_df["dist_sq"] = shots_df["dist_to_goal"]**2

    shots_df["goal_angle"] = np.abs(
        np.arctan2(Y_LEFT - shots_df["shotY"], X_GOAL - shots_df["shotX"]) -
        np.arctan2(Y_RIGHT - shots_df["shotY"], X_GOAL - shots_df["shotX"])
    )

    X = sm.add_constant(shots_df[["dist_to_goal", "dist_sq", "goal_angle"]])
    y = shots_df["shotXG"]

    model = sm.OLS(y, X).fit()
    joblib.dump(model, "shot_xg_linear_model.joblib")

# ======================================================
# 4️⃣ COMBINE → PASS GOAL PROBABILITY
# ======================================================
def compute_pass_goal_probability():
    print("4️⃣ Computing pass → goal probability")

    df = pd.read_csv("all_passes_with_future_shots_detailed.csv")

    # ---------------- Load models ----------------
    logreg = joblib.load("logreg_pass_to_shot_model.joblib")
    scaler = joblib.load("logreg_pass_to_shot_scaler.joblib")
    linreg = joblib.load("shot_xg_linear_model.joblib")

    # ---------------- Feature engineering (PASS) ----------------
    df["dist_to_goal_start"] = np.sqrt((X_GOAL - df["startX"])**2 + (Y_CENTER - df["startY"])**2)
    df["dist_to_goal_end"] = np.sqrt((X_GOAL - df["endX"])**2 + (Y_CENTER - df["endY"])**2)

    df["log_dist_to_goal_start"] = np.log1p(df["dist_to_goal_start"])
    df["log_dist_to_goal_end"] = np.log1p(df["dist_to_goal_end"])

    df["goal_angle_start"] = np.abs(
        np.arctan2(Y_LEFT - df["startY"], X_GOAL - df["startX"]) -
        np.arctan2(Y_RIGHT - df["startY"], X_GOAL - df["startX"])
    )

    df["goal_angle_end"] = np.abs(
        np.arctan2(Y_LEFT - df["endY"], X_GOAL - df["endX"]) -
        np.arctan2(Y_RIGHT - df["endY"], X_GOAL - df["endX"])
    )

    df["goal_weighted_centrality_start"] = (
        (1 - np.abs(df["startY"] - Y_CENTER) / Y_CENTER) * (df["startX"] / X_GOAL)
    )

    df["goal_weighted_centrality_end"] = (
        (1 - np.abs(df["endY"] - Y_CENTER) / Y_CENTER) * (df["endX"] / X_GOAL)
    )

    df["pass_direction"] = np.arctan2(
        df["endY"] - df["startY"],
        df["endX"] - df["startX"]
    )

    FEATURES = [
        "log_dist_to_goal_start",
        "goal_angle_start",
        "goal_weighted_centrality_start",
        "log_dist_to_goal_end",
        "goal_angle_end",
        "goal_weighted_centrality_end",
        "pass_direction"
    ]

    # ---------------- Shot probability ----------------
    X_pass = scaler.transform(df[FEATURES])
    df["shot_probability"] = logreg.predict_proba(X_pass)[:, 1]

    # ---------------- Shot → Goal (xG) ----------------
    df["max_model_xG"] = 0.0

    for i in range(1, MAX_SHOTS_PER_PASS + 1):
        mask = df[f"shotX_{i}"].notnull()
        if not mask.any():
            continue

        shotX = df.loc[mask, f"shotX_{i}"]
        shotY = df.loc[mask, f"shotY_{i}"]

        dx = X_GOAL - shotX
        dy = shotY - Y_CENTER

        dist = np.sqrt(dx**2 + dy**2)
        dist_sq = dist**2

        angle = np.abs(
            np.arctan2(Y_LEFT - shotY, X_GOAL - shotX) -
            np.arctan2(Y_RIGHT - shotY, X_GOAL - shotX)
        )

        X_shot = sm.add_constant(
            pd.DataFrame({
                "dist_to_goal": dist,
                "dist_sq": dist_sq,
                "goal_angle": angle
            })
        )

        xg_pred = linreg.predict(X_shot)

        df.loc[mask, "max_model_xG"] = np.maximum(
            df.loc[mask, "max_model_xG"],
            xg_pred
        )

    # ---------------- FINAL: Pass → Goal ----------------
    df["pass_goal_probability"] = df["shot_probability"] * df["max_model_xG"]

    df.to_csv("passes_with_goal_probabilities.csv", index=False)
    print("✔ Saved passes_with_goal_probabilities.csv")


# ======================================================
# 5️⃣ EXPECTED DANGER PER 90
# ======================================================
def rank_players_expected_danger():
    print("5️⃣ Ranking players by Expected Danger")

    # ------------------------------
    # 1️⃣ Load passes with goal probabilities
    # ------------------------------
    df = pd.read_csv("passes_with_goal_probabilities.csv")
    
    # ------------------------------
    # 2️⃣ Ensure positions are only the 5 official ones
    # ------------------------------
    official_positions = ["Central Defender", "Full Back", "Midfielder", "Striker", "Winger"]
    df = df[df["passPlayerPosition"].isin(official_positions)]
    
    # ------------------------------
    # 3️⃣ Define danger pass threshold
    # ------------------------------
    DANGER_THRESHOLD = 0.02
    df["is_danger_pass"] = df["pass_goal_probability"] >= DANGER_THRESHOLD
    
    # ------------------------------
    # 4️⃣ Aggregate per player and match (minutes counted once per match per position)
    # ------------------------------
    player_match = (
        df.groupby(
            ["passPlayerId", "passPlayerName", "passPlayerPosition", "matchId"],
            as_index=False
        )
        .agg(
            minutesPlayed=("minutesPlayed", "max"),  # per match
            total_danger=("pass_goal_probability", "sum"),
            total_danger_passes=("is_danger_pass", "sum"),
            total_passes=("passEventId", "count")
        )
    )
    
    # ------------------------------
    # 5️⃣ Aggregate across all matches per player and position
    # ------------------------------
    player_agg = (
        player_match.groupby(
            ["passPlayerId", "passPlayerName", "passPlayerPosition"],
            as_index=False
        )
        .agg(
            minutesPlayed=("minutesPlayed", "sum"),
            total_danger=("total_danger", "sum"),
            total_danger_passes=("total_danger_passes", "sum"),
            total_passes=("total_passes", "sum")
        )
    )
    
    # ------------------------------
    # 6️⃣ Compute per-90 metrics
    # ------------------------------
    player_agg["expected_danger_per_90"] = player_agg["total_danger"] / player_agg["minutesPlayed"] * 90
    player_agg["danger_passes_per_90"] = player_agg["total_danger_passes"] / player_agg["minutesPlayed"] * 90
    
    # ------------------------------
    # 7️⃣ Rename columns for clarity
    # ------------------------------
    player_agg = player_agg.rename(columns={
        "passPlayerId": "playerId",
        "passPlayerName": "playerName",
        "passPlayerPosition": "position",
    })
    
    # ------------------------------
    # 8️⃣ Sort-ready output
    # ------------------------------
    player_agg = player_agg[
        [
            "position", "playerId", "playerName", "minutesPlayed",
            "expected_danger_per_90", "danger_passes_per_90",
            "total_danger", "total_danger_passes", "total_passes"
        ]
    ]
    
    # ------------------------------
    # 9️⃣ Save each position in separate sheet + total sheet
    # ------------------------------
    position_list = official_positions
    output_file = "player_expected_danger_per_90_by_position.xlsx"
    
    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
        # Individual position sheets
        for pos in position_list:
            pos_df = player_agg[player_agg["position"] == pos]
            pos_df.to_excel(writer, sheet_name=pos, index=False)
        
        # Total sheet (sum across all positions for each player)
        total_df = (
            player_match.groupby(["passPlayerId", "passPlayerName"], as_index=False)
            .agg(
                minutesPlayed=("minutesPlayed", "sum"),  # sum across matches and positions
                total_danger=("total_danger", "sum"),
                total_danger_passes=("total_danger_passes", "sum"),
                total_passes=("total_passes", "sum")
            )
        )
        total_df["expected_danger_per_90"] = total_df["total_danger"] / total_df["minutesPlayed"] * 90
        total_df["danger_passes_per_90"] = total_df["total_danger_passes"] / total_df["minutesPlayed"] * 90
        total_df["position"] = "All"
        total_df = total_df[
            ["position", "passPlayerId", "passPlayerName", "minutesPlayed",
             "expected_danger_per_90", "danger_passes_per_90",
             "total_danger", "total_danger_passes", "total_passes"]
        ].rename(columns={"passPlayerId": "playerId", "passPlayerName": "playerName"})
        
        total_df.to_excel(writer, sheet_name="Total", index=False)
    
    print(f"Saved {output_file} with 5 position sheets plus a Total sheet.")
    print("Total players:", len(player_agg))


# ======================================================
# 
# ======================================================
if __name__ == "__main__":
    build_pass_shot_dataset()
    train_pass_to_shot_model()
    train_shot_to_goal_model()
    compute_pass_goal_probability()
    rank_players_expected_danger()
