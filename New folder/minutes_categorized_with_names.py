# -*- coding: utf-8 -*- 
"""
Created on Sun Jan 11 2026

Master list of all players with total minutes played and positions including position_provider.
"""

import pandas as pd

# ----------------------------
# 1️⃣ Load data
# ----------------------------
# Load all players
df_players = pd.read_parquet("players.parquet")
print(f"Loaded {len(df_players)} players from players.parquet")

# Create full player name (if first_name and last_name exist)
if "first_name" in df_players.columns and "last_name" in df_players.columns:
    df_players["player_name"] = df_players["first_name"] + " " + df_players["last_name"]
else:
    df_players["player_name"] = df_players["player_id"].astype(str)  # fallback

# Load minutes played
df_minutes = pd.read_parquet("minutes.parquet")
print(f"Loaded {len(df_minutes)} records from minutes.parquet")

# ----------------------------
# 2️⃣ Aggregate minutes per player per position + position_provider
# ----------------------------
position_agg = (
    df_minutes.groupby(["player_id", "position", "position_provider"], as_index=False)
              .agg(
                  minutes_played_position=("minutes_played", "sum"),
                  match_ids=("match_id", lambda x: list(sorted(x.unique())))
              )
)

# ----------------------------
# 3️⃣ Merge player info with position-wise minutes
# ----------------------------
df_master = df_players.merge(
    position_agg,
    how="left",
    on="player_id"
)

# Fill missing minutes with 0 and empty lists for match_ids
df_master["minutes_played_position"] = df_master["minutes_played_position"].fillna(0)
df_master["match_ids"] = df_master["match_ids"].apply(lambda x: x if isinstance(x, list) else [])
df_master["position_provider"] = df_master["position_provider"].fillna("Unknown")

# ----------------------------
# 4️⃣ Optional: reorder columns
# ----------------------------
columns_order = ["player_id", "player_name", "team_id", "position", "position_provider", "minutes_played_position", "match_ids"]
df_master = df_master[columns_order]

# ----------------------------
# 5️⃣ Inspect & save
# ----------------------------
print(df_master.head())
print(f"Total players (with positions expanded): {len(df_master)}")

# Save to CSV
df_master.to_csv("all_players_with_positions_minutes.csv", index=False)
print("Saved all_players_with_positions_minutes.csv")
