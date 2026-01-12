# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 16:01:12 2026

@author: hp
"""

import pandas as pd

# ----------------------------
# Load a Parquet file
# ----------------------------
file_path = "players.parquet"  # change to your file
df = pd.read_parquet(file_path)

# ----------------------------
# Inspect structure
# ----------------------------
print("===== Parquet Info =====")
print(df.info())  # shows columns, dtypes, non-null counts
print("\n===== Column Names =====")
print(df.columns.tolist())  # list all columns
print("\n===== First 5 Rows =====")
print(df.head())
# ----------------------------
# Load a Parquet file
# ----------------------------
file_path = "minutes.parquet"  # change to your file
df = pd.read_parquet(file_path)

# ----------------------------
# Inspect structure
# ----------------------------
print("===== Parquet Info =====")
print(df.info())  # shows columns, dtypes, non-null counts
print("\n===== Column Names =====")
print(df.columns.tolist())  # list all columns
print("\n===== First 5 Rows =====")
print(df.head())
