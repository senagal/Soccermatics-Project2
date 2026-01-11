import pandas as pd
import json
import os

# ----------------------------
# 1️⃣ Load minutes parquet
# ----------------------------
df_minutes = pd.read_parquet("minutes.parquet")

# ----------------------------
# 2️⃣ Aggregate minutes by player, team, position
# ----------------------------
result = (
    df_minutes.groupby(["player_id", "team_id", "position"], as_index=False)
             .agg(
                 total_minutes_played=("minutes_played", "sum"),
                 match_ids=("match_id", lambda x: sorted(x.unique()))
             )
)

# Convert match_ids list to comma-separated string
result["match_ids"] = result["match_ids"].apply(lambda x: ",".join(map(str, x)))

# Add total_matches_played column **before** match_ids
result.insert(
    result.columns.get_loc("match_ids"),
    "total_matches_played",
    result["match_ids"].apply(lambda x: len(x.split(",")))
)

# ----------------------------
# 3️⃣ Extract player names from event data
# ----------------------------
# Assuming all event JSON files are in a folder "event_data"
event_folder = "event_data"
player_names = {}

for file in os.listdir(event_folder):
    if file.endswith(".json"):
        with open(os.path.join(event_folder, file), encoding="utf-8") as f:
            data = json.load(f)
        events = data.get("events", [])
        for e in events:
            player = e.get("player")
            if player:
                pid = player.get("id")
                name = player.get("name")
                # Only add if not already in dict
                if pid not in player_names:
                    player_names[pid] = name

# ----------------------------
# 4️⃣ Map player names to minutes table
# ----------------------------
result.insert(
    1,  # Insert after player_id
    "player_name",
    result["player_id"].map(player_names)
)

# ----------------------------
# 5️⃣ Inspect & save CSV
# ----------------------------
print(result.head())

csv_path = "minutes_categorized_with_names.csv"
result.to_csv(csv_path, index=False)

print(f"\nCSV file written to: {csv_path}")
# ----------------------------
# 6️⃣ Inspect structure of events in 5588080.json
# ----------------------------
event_file = os.path.join(event_folder, "5588080.json")

with open(event_file, encoding="utf-8") as f:
    data = json.load(f)

events = data.get("events", [])

print(f"Number of events: {len(events)}\n")

# Function to recursively list structure of a dict
def list_structure(d, indent=0):
    spacing = "  " * indent
    if isinstance(d, dict):
        for k, v in d.items():
            print(f"{spacing}- {k}: {type(v).__name__}")
            if isinstance(v, dict):
                list_structure(v, indent + 1)
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                print(f"{spacing}  (list of dicts)")
                list_structure(v[0], indent + 2)
    elif isinstance(d, list):
        if d and isinstance(d[0], dict):
            list_structure(d[0], indent)
        else:
            print(f"{spacing}- list of {type(d[0]).__name__} (length {len(d)})" if d else f"{spacing}- empty list")

# List structure of the first event (representative)
if events:
    print("Structure of first event:")
    list_structure(events[0])
else:
    print("No events found in the file.")
# ----------------------------
# 7️⃣ Fetch and print at least one event
# ----------------------------
if events:
    first_event = events[0]  # grab the first event
    print("\nFirst event data (sample):")
    print(json.dumps(first_event, indent=2))  # pretty-print the JSON
else:
    print("No events found in the file.")
