# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 16:31:03 2026

@author: senaa
"""

import json
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import statsmodels.formula.api as smf

# ------------------------------
# 1. Directory containing all JSON event files
# ------------------------------
EVENTS_DIR = "event_data"

all_passes = []

# ------------------------------
# 2. Loop through all JSON files and extract passes + shots
# ------------------------------
for filename in os.listdir(EVENTS_DIR):
    if not filename.endswith(".json"):
        continue

    file_path = os.path.join(EVENTS_DIR, filename)

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    events = data.get("events", [])
    if not events:
        continue
    match_id = events[0].get("matchId")

    passes, shots = [], []

    for event in events:
        minute, second = event.get("minute",0), event.get("second",0)
        time_seconds = minute*60 + second

        # -------- Shots --------
        if event.get("shot") is not None:
            shots.append({
                "shotEventId": event.get("id"),
                "shotTeamId": event.get("team", {}).get("id"),
                "shotTimeSeconds": time_seconds,
                "shotPlayerId": event.get("player", {}).get("id"),
                "shotPlayerName": event.get("player", {}).get("name"),
                "startX": event.get("location", {}).get("x"),
                "startY": event.get("location", {}).get("y"),
                "isGoal": int(event.get("shot", {}).get("isGoal", False))
            })

        # -------- Passes --------
        if event.get("pass") is not None:
            passes.append({
                "matchId": match_id,
                "passEventId": event.get("id"),
                "passTeamId": event.get("team", {}).get("id"),
                "passPlayerId": event.get("player", {}).get("id"),
                "passTimeSeconds": time_seconds,
                "startX": event.get("location", {}).get("x"),
                "startY": event.get("location", {}).get("y"),
                "endX": event.get("pass", {}).get("endLocation", {}).get("x"),
                "endY": event.get("pass", {}).get("endLocation", {}).get("y"),
                "becameShot": False,
                "shotEventId": None,
                "shotOutcome": None,
                "shotPlayerId": None,
                "shotPlayerName": None,
                "shotStartX": None,
                "shotStartY": None,
                "isGoal": None
            })

    if not passes or not shots:
        continue

    # Convert to DataFrames
    shots_df = pd.DataFrame(shots)
    passes_df = pd.DataFrame(passes)

    # ------------------------------
    # 3. Mark last pass within 15 seconds before each shot
    # ------------------------------
    for _, shot in shots_df.iterrows():
        shot_time, team_id = shot["shotTimeSeconds"], shot["shotTeamId"]
        relevant_passes = passes_df[
            (passes_df["passTeamId"]==team_id) &
            (passes_df["passTimeSeconds"] >= shot_time-15) &
            (passes_df["passTimeSeconds"] < shot_time)
        ]
        if relevant_passes.empty:
            continue
        last_pass_idx = relevant_passes["passTimeSeconds"].idxmax()
        passes_df.at[last_pass_idx, "becameShot"] = True
        passes_df.at[last_pass_idx, "shotEventId"] = shot["shotEventId"]
        passes_df.at[last_pass_idx, "shotOutcome"] = "Goal" if shot["isGoal"] else "No Goal"
        passes_df.at[last_pass_idx, "shotPlayerId"] = shot["shotPlayerId"]
        passes_df.at[last_pass_idx, "shotPlayerName"] = shot["shotPlayerName"]
        passes_df.at[last_pass_idx, "shotStartX"] = shot["startX"]
        passes_df.at[last_pass_idx, "shotStartY"] = shot["startY"]
        passes_df.at[last_pass_idx, "isGoal"] = shot["isGoal"]

    all_passes.extend(passes_df.to_dict("records"))

# ------------------------------
# 4. Final DataFrame
# ------------------------------
df_passes = pd.DataFrame(all_passes)

# ------------------------------
# 5. Compute shot features (for goal regression)
# ------------------------------
goal_x, goal_y_left, goal_y_right = 100, 37, 63
df_passes['shot_dist_to_goal'] = np.sqrt((goal_x - df_passes['shotStartX'])**2 +
                                        (50 - df_passes['shotStartY'])**2)
angle_left  = np.arctan2(goal_y_left - df_passes['shotStartY'], goal_x - df_passes['shotStartX'])
angle_right = np.arctan2(goal_y_right - df_passes['shotStartY'], goal_x - df_passes['shotStartX'])
df_passes['shot_goal_angle'] = np.abs(angle_right - angle_left)

# ------------------------------
# 6. Pass → Shot probability (logistic regression)
# ------------------------------
features_pass = ["startX","startY","endX","endY","shot_dist_to_goal","shot_goal_angle"]
df_passes_for_model = df_passes.fillna(0)  # fill NaN for passes without shot

X_pass = df_passes_for_model[features_pass]
y_pass = df_passes_for_model["becameShot"].astype(int)

scaler = StandardScaler()
X_pass_scaled = scaler.fit_transform(X_pass)

model_pass = LogisticRegression(max_iter=1000)
model_pass.fit(X_pass_scaled, y_pass)

df_passes['passShotProbability'] = model_pass.predict_proba(scaler.transform(X_pass))[:,1]

# ------------------------------
# 7. Shot → Goal probability (linear regression)
# ------------------------------
df_shot_passes = df_passes[df_passes["becameShot"]].copy()

model_goal = smf.ols(formula='isGoal ~ shotStartX + shotStartY + shot_dist_to_goal + shot_goal_angle', 
                     data=df_shot_passes).fit()
df_shot_passes['goalProbability'] = model_goal.predict(df_shot_passes).clip(0,1)
# Save the model for later use
model_goal.save("shot_goal_model.pickle")
print("Linear regression model saved as shot_goal_model.pickle")
# Show full summary
print(model_goal.summary())

# ------------------------------
# 8. Combine probabilities for all passes
# ------------------------------
df_passes = df_passes.merge(df_shot_passes[['passEventId','goalProbability']], 
                            on='passEventId', how='left')
df_passes['goalProbability'] = df_passes['goalProbability'].fillna(0)

# ------------------------------
# 9. Pass → Goal probability
# ------------------------------
df_passes['passGoalProbability'] = df_passes['passShotProbability'] * df_passes['goalProbability']

