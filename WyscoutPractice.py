import json
import pandas as pd
import numpy as np
import os
import pandas as pd

file_path = os.path.join("event_data", "5588080.json")

with open(file_path, encoding="utf-8") as f:
    data = json.load(f)
df = pd.DataFrame(data['events'])
pd.set_option('display.max_colwidth', None)
print(df.head())
print(df.columns)