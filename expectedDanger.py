# -*- coding: utf-8 -*-    
"""
Player ranking by Expected Danger per 90 using corrected passes_with_goal_probabilities.csv
Aggregates per match to count minutes only once, then computes per-90 metrics.
Saves each position in a separate sheet plus a total sheet.
"""

import pandas as pd

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
