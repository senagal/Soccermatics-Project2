# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 10:15:55 2026

@author: senaa
"""

import json
import pandas as pd
import os
import numpy as np

EVENTS_DIR = "event_data"

all_passes = []

# ----------------------------------
# LOOP THROUGH ALL JSON FILES
# ----------------------------------

for filename in os.listdir(EVENTS_DIR):

    if not filename.endswith(".json"):
        continue

    file_path = os.path.join(EVENTS_DIR, filename)

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    events = data.get("events", [])
    match_id = events[0].get("matchId") if events else None

    shots = []
    passes = []

    # ----------------------------------
    # STEP 1: Split passes and shots
    # ----------------------------------

    for event in events:

        minute = event.get("minute", 0)
        second = event.get("second", 0)
        time_seconds = minute * 60 + second

        # -------- SHOTS --------
        if event.get("shot") is not None:
            shots.append({
                "shotEventId": event.get("id"),
                "shotTeamId": event.get("team", {}).get("id"),
                "shotTimeSeconds": time_seconds,
            })

        # -------- PASSES --------
        if event.get("pass") is not None:
            passes.append({
                "matchId": match_id,
                "passEventId": event.get("id"),
                "passTeamId": event.get("team", {}).get("id"),
                "passPlayerId": event.get("player", {}).get("id"),
                "passTimeSeconds": time_seconds,

                # PASS COORDINATES
                "startX": event.get("location", {}).get("x"),
                "startY": event.get("location", {}).get("y"),
                "endX": event.get("pass", {}).get("endLocation", {}).get("x"),
                "endY": event.get("pass", {}).get("endLocation", {}).get("y"),

                # TARGET VARIABLE
                "becameShot": False
            })

    if not shots or not passes:
        all_passes.extend(passes)
        continue

    shots_df = pd.DataFrame(shots)
    passes_df = pd.DataFrame(passes)

    # ----------------------------------
    # STEP 2: Mark last pass within 15s before each shot
    # ----------------------------------

    for _, shot in shots_df.iterrows():

        shot_time = shot["shotTimeSeconds"]
        team_id = shot["shotTeamId"]

        window_start = shot_time - 15

        relevant_passes = passes_df[
            (passes_df["passTeamId"] == team_id) &
            (passes_df["passTimeSeconds"] >= window_start) &
            (passes_df["passTimeSeconds"] < shot_time)
        ]

        if relevant_passes.empty:
            continue

        # last pass before the shot
        last_pass_idx = relevant_passes["passTimeSeconds"].idxmax()
        passes_df.at[last_pass_idx, "becameShot"] = True

    # ----------------------------------
    # STEP 3: Calculate features
    # ----------------------------------

    goal_x, goal_y_left, goal_y_right = 100, 37, 63  # Wyscout coordinates for goal posts

    # Distance from goal (Euclidean distance to goal center)
    passes_df["dist_to_goal"] = np.sqrt((100 - passes_df["endX"])**2 + (50 - passes_df["endY"])**2)
    passes_df["log_distance_to_goal"] = np.log1p(passes_df["dist_to_goal"])  # log(d+1)

    # Goal angle
    angle_left  = np.arctan2(goal_y_left - passes_df["endY"], goal_x - passes_df["endX"])
    angle_right = np.arctan2(goal_y_right - passes_df["endY"], goal_x - passes_df["endX"])
    passes_df["goal_angle"] = np.abs(angle_right - angle_left)

    # collect passes from this match
    all_passes.extend(passes_df.to_dict("records"))

# ----------------------------------
# OUTPUT
# ----------------------------------

passes_df_final = pd.DataFrame(all_passes)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

print(passes_df_final.head())
print("\nTotal passes:", len(passes_df_final))
print("Passes that became shots:", passes_df_final["becameShot"].sum())

# Save to CSV
passes_df_final.to_csv("all_passes_with_features.csv", index=False)
print("\nSaved all_passes_with_features.csv")
