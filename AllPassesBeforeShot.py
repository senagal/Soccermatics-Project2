# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 11:31:59 2026

@author: senaa
"""

import json
import pandas as pd

file_path = "event_data/5588080.json"

with open(file_path, encoding="utf-8") as f:
    data = json.load(f)

events = data["events"]

# ----------------------------------
# STEP 1: Split passes and shots
# ----------------------------------

shots = []
passes = []

for event in events:

    minute = event.get("minute", 0)
    second = event.get("second", 0)
    time_seconds = minute * 60 + second

    # -------- SHOTS --------
    if event.get("shot") is not None:
        shots.append({
            "shotEventId": event.get("id"),
            "shotTeamId": event.get("team", {}).get("id"),
            "shotPlayerId": event.get("player", {}).get("id"),
            "shotTimeSeconds": time_seconds,
            "shotTime": f"{minute:02d}:{second:02d}",
        })

    # -------- PASSES --------
    if event.get("pass") is not None:
        passes.append({
            "passEventId": event.get("id"),
            "passTeamId": event.get("team", {}).get("id"),
            "passPlayerId": event.get("player", {}).get("id"),
            "passTimeSeconds": time_seconds,
            "passTime": f"{minute:02d}:{second:02d}",

            # PASS COORDINATES ONLY
            "startX": event.get("location", {}).get("x"),
            "startY": event.get("location", {}).get("y"),
            "endX": event.get("pass", {}).get("endLocation", {}).get("x"),
            "endY": event.get("pass", {}).get("endLocation", {}).get("y"),
        })

shots_df = pd.DataFrame(shots)
passes_df = pd.DataFrame(passes)

# ----------------------------------
# STEP 2: Last pass within 15s BEFORE each shot
# ----------------------------------

danger_passes = []

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

    for _, p in relevant_passes.iterrows():
        danger_passes.append({
            # SHOT INFO
            "shotEventId": shot["shotEventId"],
            "shotTeamId": shot["shotTeamId"],
            "shotPlayerId": shot["shotPlayerId"],
            "shotTime": shot["shotTime"],
    
            # PASS INFO
            "passEventId": p["passEventId"],
            "passPlayerId": p["passPlayerId"],
            "passTime": p["passTime"],
            "secondsBeforeShot": shot["shotTimeSeconds"] - p["passTimeSeconds"],
    
            # PASS COORDINATES
            "startX": p["startX"],
            "startY": p["startY"],
            "endX": p["endX"],
            "endY": p["endY"],
        })


danger_passes_df = pd.DataFrame(danger_passes)

# ----------------------------------
# OUTPUT
# ----------------------------------

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

print(danger_passes_df.head())
print("\nTotal danger passes:", len(danger_passes_df))

danger_passes_df.to_csv("All_passes_before_shot.csv", index=False)
print("\nSaved passes_last_before_shot.csv")
