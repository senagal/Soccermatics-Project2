import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file = "player_expected_danger_per_90_by_position.xlsx"
xls = pd.ExcelFile(file)

# Only the 5 positions, ignore the Total sheet
position_sheets = [s for s in xls.sheet_names if s != "Total"]

# Combine all position sheets into one DataFrame
position_data = []
for pos in position_sheets:
    df_pos = xls.parse(pos)
    df_pos["Position"] = pos  # add a column for position

    # Strip strings, lowercase, replace 'inf' with NaN, then convert
    for col in ["expected_danger_per_90", "danger_passes_per_90"]:
        df_pos[col] = df_pos[col].astype(str).str.strip().str.lower()
        df_pos[col] = df_pos[col].replace("inf", np.nan)
        df_pos[col] = pd.to_numeric(df_pos[col], errors="coerce")

    position_data.append(df_pos)

df_all_positions = pd.concat(position_data, ignore_index=True)

# -----------------------------
# Quick summary per position
# -----------------------------
summary = df_all_positions.groupby("Position").agg(
    Avg_xD_per_90=("expected_danger_per_90", "mean"),
    Avg_danger_passes_per_90=("danger_passes_per_90", "mean"),
    Num_players=("playerId", "nunique")
).reset_index()

print(summary)

# -----------------------------
# Barplot: Expected Danger per 90
# -----------------------------
plt.figure(figsize=(8,5))
sns.barplot(data=summary, x="Position", y="Avg_xD_per_90", palette="viridis")
plt.ylabel("Avg Expected Danger per 90")
plt.title("Expected Danger per 90 by Position")
plt.show()

# -----------------------------
# Barplot: Danger Passes per 90
# -----------------------------
plt.figure(figsize=(8,5))
sns.barplot(data=summary, x="Position", y="Avg_danger_passes_per_90", palette="magma")
plt.ylabel("Avg Danger Passes per 90")
plt.title("Danger Passes per 90 by Position")
plt.show()

# -----------------------------
# Scatterplot: xD vs Danger Passes per 90
# -----------------------------
plt.figure(figsize=(7,5))
sns.scatterplot(
    data=summary,
    x="Avg_danger_passes_per_90",
    y="Avg_xD_per_90",
    hue="Position",
    s=200
)
for i, row in summary.iterrows():
    plt.text(row["Avg_danger_passes_per_90"]+0.01, row["Avg_xD_per_90"], row["Position"])
plt.xlabel("Avg Danger Passes per 90")
plt.ylabel("Avg Expected Danger per 90")
plt.title("Position Effect on Expected Danger")
plt.grid(True)
plt.show()
