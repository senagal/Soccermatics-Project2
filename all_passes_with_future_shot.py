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

# ----------------------------------
# LOAD MINUTES DATA
# ----------------------------------
minutes_df = pd.read_parquet("minutes.parquet")

# Keep only required columns
minutes_df = minutes_df[["player_id", "match_id", "minutes_played"]]

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

                # ✅ PLAYER INFO FROM EVENT DATA
                "passPlayerName": event.get("player", {}).get("name"),
                "passPlayerPosition": event.get("player", {}).get("position"),

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

            for i, (_, shot) in enumerate(
                future_shots.head(MAX_SHOTS_PER_PASS).iterrows()
            ):
                j = i + 1
                passes_df.at[idx, f"shotId_{j}"] = shot["shotEventId"]
                passes_df.at[idx, f"shotX_{j}"] = shot["shotX"]
                passes_df.at[idx, f"shotY_{j}"] = shot["shotY"]
                passes_df.at[idx, f"shotIsGoal_{j}"] = shot["shotIsGoal"]
                passes_df.at[idx, f"shotXG_{j}"] = shot["shotXG"]

    all_passes.extend(passes_df.to_dict("records"))

# ----------------------------------
# FINAL DATAFRAME
# ----------------------------------
passes_df_final = pd.DataFrame(all_passes)

# ----------------------------------
# MAP POSITION ACRONYMS TO OFFICIAL POSITIONS
# ----------------------------------
position_map = {
    # Central Defender
    "CB": "Central Defender", "LCB": "Central Defender", "LCB3": "Central Defender",
    "RCB": "Central Defender", "RCB3": "Central Defender",

    # Full Back
    "LB": "Full Back", "LB5": "Full Back", "LWB": "Full Back",
    "RB": "Full Back", "RB5": "Full Back", "RWB": "Full Back",

    # Midfielder
    "AMF": "Midfielder", "DMF": "Midfielder", "LCMF": "Midfielder",
    "LCMF3": "Midfielder", "LDMF": "Midfielder", "RCMF": "Midfielder",
    "RCMF3": "Midfielder", "RDMF": "Midfielder",

    # Striker
    "CF": "Striker", "SS": "Striker",

    # Winger
    "LAMF": "Winger", "LW": "Winger", "LWF": "Winger",
    "RAMF": "Winger", "RW": "Winger", "RWF": "Winger"
}

# Replace acronyms with official positions
passes_df_final["passPlayerPosition"] = passes_df_final["passPlayerPosition"].map(position_map).fillna("Unknown")

# Optional: store the original provider acronym
passes_df_final["position_provider"] = passes_df_final["passPlayerPosition"]

# ----------------------------------
# MERGE MINUTES PLAYED (MATCH + PLAYER)
# ----------------------------------

# Keep only one row per player per match
minutes_unique = minutes_df.drop_duplicates(subset=["player_id", "match_id"])

passes_df_final = passes_df_final.merge(
    minutes_unique,
    how="left",
    left_on=["passPlayerId", "matchId"],
    right_on=["player_id", "match_id"]
)

passes_df_final = passes_df_final.rename(
    columns={"minutes_played": "minutesPlayed"}
)

passes_df_final = passes_df_final.drop(
    columns=["player_id", "match_id"]
)

# ----------------------------------
# OUTPUT
# ----------------------------------
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

print(passes_df_final.head())
print("\nTotal passes:", len(passes_df_final))
print("Passes that led to shots:", passes_df_final["becameShot"].sum())

passes_df_final.to_csv(
    "all_passes_with_future_shots_detailed.csv",
    index=False
)

print("\nSaved all_passes_with_future_shots_detailed.csv")
