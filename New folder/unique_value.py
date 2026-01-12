# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 19:02:57 2026

@author: hp
"""

import pandas as pd

# Load your CSV file
df = pd.read_csv("all_players_with_positions_minutes.csv")

# Option 1: If the column header is literally named "F"
unique_count = df['position'].nunique()
print("Unique values in column F (by header):", unique_count)

# Option 2: If you mean the 6th column (Excel column F by position)
unique_count_pos = df.iloc[:, 5].nunique()
print("Unique values in 6th column (Excel F):", unique_count_pos)
unique_values = df.iloc[:, 5].unique()
print("Unique values in column F:", unique_values)
value_counts = df.iloc[:, 5].value_counts()
print(value_counts)
