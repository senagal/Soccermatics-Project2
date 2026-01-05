# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 16:45:03 2026
@author: senaa
"""

import pandas as pd

# Load parquet file
df = pd.read_parquet("players.parquet")

# Select columns by index
cols = [0, 1, 2, 3, 10, 13]
players_subset = df.iloc[:, cols]

# Save to CSV
players_subset.to_csv("players_reference.csv", index=False)

print("File saved as players_reference.csv")
