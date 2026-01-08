import json
import pprint

file_path = "event_data/5588080.json"

# Load JSON
with open(file_path, encoding="utf-8") as f:
    data = json.load(f)

events = data.get("events", [])
print(f"Number of events in this file: {len(events)}\n")

# Find first shot event
shot_event = None
for event in events:
    if event.get("shot") is not None:
        shot_event = event
        break

if shot_event:
    print("Keys in a shot event:")
    print(list(shot_event.keys()))
    print("\nFull shot event structure (pretty-printed):")
    pprint.pprint(shot_event)
    
    print("\nKeys inside the 'shot' field:")
    print(list(shot_event['shot'].keys()))
    print("\n'outcome' subfield in shot (if exists):")
    if shot_event['shot'].get("outcome"):
        pprint.pprint(shot_event['shot']['outcome'])
    else:
        print("No outcome field found in this shot")
    
    print("\n'subtype' subfield in shot (if exists):")
    if shot_event['shot'].get("subtype"):
        pprint.pprint(shot_event['shot']['subtype'])
    else:
        print("No subtype field found in this shot")
else:
    print("No shot events found in this file.")
