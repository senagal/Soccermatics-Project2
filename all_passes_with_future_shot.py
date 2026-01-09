# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 15:33:22 2026

@author: senaa
"""

import json
import pandas as pd
import os

EVENTS_DIR = "event_data"
SHOT_WINDOW = 15
MAX_SHOTS_PER_PASS = 4

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

    passes = []
    shots = []

    # ----------------------------------
    # EXTRACT PASSES AND SHOTS
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

                # SHOT FEATURES
                "shotX": event.get("location", {}).get("x"),
                "shotY": event.get("location", {}).get("y"),
                "shotIsGoal": event.get("shot", {}).get("isGoal"),
                "shotXG": event.get("shot", {}).get("xg"),
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

                # TARGET
                "becameShot": False,

                # SHOT PLACEHOLDERS
                **{
                    f"{field}_{i+1}": None
                    for i in range(MAX_SHOTS_PER_PASS)
                    for field in [
                        "shotId",
                        "shotX",
                        "shotY",
                        "shotIsGoal",
                        "shotXG",
                    ]
                }
            })

    if not passes:
        continue

    passes_df = pd.DataFrame(passes)
    shots_df = pd.DataFrame(shots)

    # ----------------------------------
    # LINK PASSES TO FUTURE SHOTS
    # ----------------------------------

    if not shots_df.empty:

        for idx, p in passes_df.iterrows():

            pass_time = p["passTimeSeconds"]
            team_id = p["passTeamId"]

            window_end = pass_time + SHOT_WINDOW

            future_shots = shots_df[
                (shots_df["shotTeamId"] == team_id) &
                (shots_df["shotTimeSeconds"] > pass_time) &
                (shots_df["shotTimeSeconds"] <= window_end)
            ].sort_values("shotTimeSeconds")

            if future_shots.empty:
                continue

            passes_df.at[idx, "becameShot"] = True

            for i, (_, shot) in enumerate(future_shots.head(MAX_SHOTS_PER_PASS).iterrows()):
                j = i + 1

                passes_df.at[idx, f"shotId_{j}"] = shot["shotEventId"]
                passes_df.at[idx, f"shotX_{j}"] = shot["shotX"]
                passes_df.at[idx, f"shotY_{j}"] = shot["shotY"]
                passes_df.at[idx, f"shotIsGoal_{j}"] = shot["shotIsGoal"]
                passes_df.at[idx, f"shotXG_{j}"] = shot["shotXG"]

    all_passes.extend(passes_df.to_dict("records"))

# ----------------------------------
# OUTPUT
# ----------------------------------

passes_df_final = pd.DataFrame(all_passes)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

print(passes_df_final.head())
print("\nTotal passes:", len(passes_df_final))
print("Passes that led to shots:", passes_df_final["becameShot"].sum())

passes_df_final.to_csv("all_passes_with_future_shots_detailed.csv", index=False)
print("\nSaved all_passes_with_future_shots_detailed.csv")
