# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 14:34:14 2026

@author: senaa
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 2026
@author: senaa
"""

import pandas as pd

# Path to your parquet file
file_path = "matches.parquet"

# Load the parquet file
matches_df = pd.read_parquet(file_path)

# Optional: show all columns and rows
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", None)

# Display first 10 rows
print(matches_df.head(10))

# If you want to display the full DataFrame (careful if it's very large):
# print(matches_df)
