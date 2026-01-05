# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 11:37:29 2026

@author: senaa
"""

import json
import pandas as pd
import os

EVENTS_DIR = "event_data"

all_danger_passes = []

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

                # PASS COORDINATES
                "startX": event.get("location", {}).get("x"),
                "startY": event.get("location", {}).get("y"),
                "endX": event.get("pass", {}).get("endLocation", {}).get("x"),
                "endY": event.get("pass", {}).get("endLocation", {}).get("y"),
            })

    if not shots or not passes:
        continue

    shots_df = pd.DataFrame(shots)
    passes_df = pd.DataFrame(passes)

    # ----------------------------------
    # STEP 2: Last pass within 15s BEFORE each shot
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

        last_pass = relevant_passes.loc[
            relevant_passes["passTimeSeconds"].idxmax()
        ]

        all_danger_passes.append({
            "matchId": match_id,

            # SHOT INFO
            "shotEventId": shot["shotEventId"],
            "shotTeamId": shot["shotTeamId"],
            "shotPlayerId": shot["shotPlayerId"],
            "shotTime": shot["shotTime"],

            # PASS INFO
            "passEventId": last_pass["passEventId"],
            "passPlayerId": last_pass["passPlayerId"],
            "passTime": last_pass["passTime"],
            "secondsBeforeShot": shot_time - last_pass["passTimeSeconds"],

            # PASS COORDINATES
            "startX": last_pass["startX"],
            "startY": last_pass["startY"],
            "endX": last_pass["endX"],
            "endY": last_pass["endY"],
        })

# ----------------------------------
# OUTPUT
# ----------------------------------

danger_passes_df = pd.DataFrame(all_danger_passes)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

print(danger_passes_df.head())
print("\nTotal danger passes:", len(danger_passes_df))

danger_passes_df.to_csv("passes_last_before_shot_ALL_MATCHES.csv", index=False)
print("\nSaved passes_last_before_shot_ALL_MATCHES.csv")
