import json
import pandas as pd

file_path = "event_data/5588080.json"

with open(file_path, encoding="utf-8") as f:
    data = json.load(f)

events = data["events"]

shots_rows = []

for event in events:
    if event.get("shot") is not None:

        location = event.get("location", {})

        row = {
            "eventId": event.get("id"),
            "matchId": event.get("matchId"),
            "period": event.get("matchPeriod"),
            "minute": event.get("minute"),
            "second": event.get("second"),
            "teamId": event.get("team", {}).get("id"),
            "playerId": event.get("player", {}).get("id"),
            "x": location.get("x"),
            "y": location.get("y"),
        }

        shot = event["shot"]

        row["isGoal"] = shot.get("isGoal")

        # Robust bodyPart extraction
        body_part = shot.get("bodyPart")
        if isinstance(body_part, dict):
            row["bodyPart"] = body_part.get("name")
        else:
            row["bodyPart"] = body_part  # might already be a string or None

        # Robust shotType extraction
        shot_type = shot.get("type")
        if isinstance(shot_type, dict):
            row["shotType"] = shot_type.get("name")
        else:
            row["shotType"] = shot_type

        shots_rows.append(row)

# Create DataFrame
shots_df = pd.DataFrame(shots_rows)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

print(shots_df.head())
print("\nTotal shots:", len(shots_df))

# Save to CSV
shots_df.to_csv("match_5588080_shots.csv", index=False)
print("\nSaved match_5588080_shots.csv")
