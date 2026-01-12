# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 2026

Compute shot-based xG per pass using Linear Regression (target = shotXG)
Only keeps the highest-xG shot per pass
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import matplotlib.pyplot as plt

# ------------------------------
# 1️⃣ Load dataset
# ------------------------------
df = pd.read_csv("all_passes_with_future_shots_detailed.csv")

# Keep only passes that became shots
df = df[df["becameShot"] == True].copy()

MAX_SHOTS_PER_PASS = 4

# ------------------------------
# 2️⃣ Collect all shots and prepare features
# ------------------------------
GOAL_X = 100
GOAL_Y = 50
Y_CENTER = 50
Y_LEFT, Y_RIGHT = 37, 63

all_shots = []

for i in range(1, MAX_SHOTS_PER_PASS + 1):
    mask = df[f"shotX_{i}"].notnull()
    tmp = df.loc[mask, [
        "matchId", "passEventId",
        "startX", "startY",
        f"shotX_{i}", f"shotY_{i}", f"shotXG_{i}"
    ]].copy()
    
    tmp = tmp.rename(columns={
        "startX": "shotStartX",
        "startY": "shotStartY",
        f"shotX_{i}": "shotX",
        f"shotY_{i}": "shotY",
        f"shotXG_{i}": "shotXG"
    })
    tmp["shotNumber"] = i
    all_shots.append(tmp)

shots_df = pd.concat(all_shots, ignore_index=True)

# ------------------------------
# 3️⃣ Feature engineering
# ------------------------------
# Shot START features
dx_start = GOAL_X - shots_df["shotStartX"]
dy_start = shots_df["shotStartY"] - GOAL_Y
shots_df["dist_to_goal_start"] = np.sqrt(dx_start**2 + dy_start**2)
shots_df["dist_to_goal_start_sq"] = shots_df["dist_to_goal_start"]**2
angle_left_start = np.arctan2(Y_LEFT - shots_df["shotStartY"], GOAL_X - shots_df["shotStartX"])
angle_right_start = np.arctan2(Y_RIGHT - shots_df["shotStartY"], GOAL_X - shots_df["shotStartX"])
shots_df["goal_angle_start"] = np.abs(angle_right_start - angle_left_start)
shots_df["goal_weighted_centrality_start"] = 1 - (np.abs(dy_start)/50)

# Shot END features
dx_end = GOAL_X - shots_df["shotX"]
dy_end = shots_df["shotY"] - GOAL_Y
shots_df["dist_to_goal_end"] = np.sqrt(dx_end**2 + dy_end**2)
shots_df["dist_to_goal_end_sq"] = shots_df["dist_to_goal_end"]**2
angle_left_end = np.arctan2(Y_LEFT - shots_df["shotY"], GOAL_X - shots_df["shotX"])
angle_right_end = np.arctan2(Y_RIGHT - shots_df["shotY"], GOAL_X - shots_df["shotX"])
shots_df["goal_angle_end"] = np.abs(angle_right_end - angle_left_end)
shots_df["goal_weighted_centrality_end"] = 1 - (np.abs(dy_end)/50)

# ------------------------------
# 4️⃣ Clean target variable
# ------------------------------
shots_df = shots_df[shots_df["shotXG"].notnull()].copy()
y = shots_df["shotXG"]

# ------------------------------
# 5️⃣ Fit linear regression xG model
# ------------------------------
features = [
    "dist_to_goal_start", "dist_to_goal_start_sq", "goal_angle_start", "goal_weighted_centrality_start",
    "dist_to_goal_end", "dist_to_goal_end_sq", "goal_angle_end", "goal_weighted_centrality_end"
]
X = shots_df[features]

model = LinearRegression()
model.fit(X, y)
shots_df["model_xG"] = model.predict(X)

# ------------------------------
# 6️⃣ Keep only the highest-xG shot per pass
# ------------------------------
shots_df_max = shots_df.loc[shots_df.groupby("passEventId")["model_xG"].idxmax()].copy()

# ------------------------------
# 7️⃣ Map max xG back to passes
# ------------------------------
df = df.merge(
    shots_df_max[["matchId", "passEventId", "model_xG"]].rename(columns={"model_xG": "max_model_xG"}),
    on=["matchId", "passEventId"],
    how="left"
)

# ------------------------------
# 8️⃣ Model summary
# ------------------------------
print("\n===== xG Model Summary =====")
print(f"Number of passes: {len(df)}")
print(f"Number of shots: {len(shots_df)}")
print(f"Mean predicted max xG per pass: {shots_df_max['model_xG'].mean():.4f}")
print(f"Std of predicted max xG per pass: {shots_df_max['model_xG'].std():.4f}")
print("\nLinear Regression Coefficients:")
for feat, coef in zip(features, model.coef_):
    print(f"  {feat}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
r2 = model.score(X, y)
print(f"R^2 score: {r2:.4f}")
print("============================\n")

# ------------------------------
# 9️⃣ Save results
# ------------------------------
df.to_csv("passes_with_max_model_xG.csv", index=False)
joblib.dump(model, "shot_xg_linear_model.joblib")
print("Saved passes_with_max_model_xG.csv and linear model.")

# ------------------------------
# 10️⃣ Optional: Plot effect of pass start/end X on xG
# ------------------------------
def plot_whole_numbers(x_vals, y_vals, xlabel="X", title="Effect on xG", color="blue"):
    df_plot = pd.DataFrame({"x": x_vals, "xg": y_vals})
    df_plot = df_plot[df_plot["x"] % 1 == 0]  # whole numbers
    if df_plot.empty:
        return
    x_plot = df_plot["x"].values.reshape(-1,1)
    y_plot = df_plot["xg"].values
    lr = LinearRegression()
    lr.fit(x_plot, y_plot)
    y_fit = lr.predict(x_plot)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.scatter(x_plot, y_plot, alpha=0.7, s=40, color=color)
    ax.plot(x_plot, y_fit, color='red', linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Average Shot xG")
    ax.set_title(title)
    ax.grid(True)
    plt.show()

mean_xg_per_pass = df["max_model_xG"].values
plot_whole_numbers(df["startX"].values, mean_xg_per_pass, xlabel="Pass Start X", title="Effect of Pass Start X on Shot xG", color="blue")
plot_whole_numbers(df["endX"].values, mean_xg_per_pass, xlabel="Pass End X", title="Effect of Pass End X on Shot xG", color="green")
