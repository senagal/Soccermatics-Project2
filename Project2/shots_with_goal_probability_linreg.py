# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 17:07:52 2026

@author: hp
"""

import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm

# ------------------------------
# 1. Load dataset
# ------------------------------
df = pd.read_csv("all_passes_with_future_shots_detailed.csv")

MAX_SHOTS_PER_PASS = 4

# Wyscout pitch constants
X_GOAL = 100
Y_CENTER = 50
Y_LEFT, Y_RIGHT = 37, 63

# ------------------------------
# 2. Keep only passes that led to shots
# ------------------------------
df_shots = df[df["becameShot"] == True].copy()

all_shots = []

for i in range(1, MAX_SHOTS_PER_PASS + 1):
    mask = df_shots[f"shotX_{i}"].notnull()

    tmp = df_shots.loc[
        mask,
        [
            "matchId",
            "passEventId",
            "startX",
            "startY",
            f"shotX_{i}",
            f"shotY_{i}",
            f"shotXG_{i}",
        ],
    ].copy()

    tmp = tmp.rename(
        columns={
            "startX": "shotStartX",
            "startY": "shotStartY",
            f"shotX_{i}": "shotX",
            f"shotY_{i}": "shotY",
            f"shotXG_{i}": "shotXG",
        }
    )

    tmp["shotNumber"] = i
    all_shots.append(tmp)

shots_df = pd.concat(all_shots, ignore_index=True)
shots_df = shots_df[shots_df["shotXG"].notnull()].copy()

print(f"Total shots used for xG model: {len(shots_df)}")

# ------------------------------
# 3. Feature engineering
# ------------------------------
# Start location
dx_start = X_GOAL - shots_df["shotStartX"]
dy_start = shots_df["shotStartY"] - Y_CENTER

shots_df["dist_to_goal_start"] = np.sqrt(dx_start**2 + dy_start**2)
shots_df["dist_to_goal_start_sq"] = shots_df["dist_to_goal_start"] ** 2

angle_left_start = np.arctan2(Y_LEFT - shots_df["shotStartY"], X_GOAL - shots_df["shotStartX"])
angle_right_start = np.arctan2(Y_RIGHT - shots_df["shotStartY"], X_GOAL - shots_df["shotStartX"])
shots_df["goal_angle_start"] = np.abs(angle_right_start - angle_left_start)

shots_df["goal_weighted_centrality_start"] = 1 - np.abs(dy_start) / Y_CENTER

# End location
dx_end = X_GOAL - shots_df["shotX"]
dy_end = shots_df["shotY"] - Y_CENTER

shots_df["dist_to_goal_end"] = np.sqrt(dx_end**2 + dy_end**2)
shots_df["dist_to_goal_end_sq"] = shots_df["dist_to_goal_end"] ** 2

angle_left_end = np.arctan2(Y_LEFT - shots_df["shotY"], X_GOAL - shots_df["shotX"])
angle_right_end = np.arctan2(Y_RIGHT - shots_df["shotY"], X_GOAL - shots_df["shotX"])
shots_df["goal_angle_end"] = np.abs(angle_right_end - angle_left_end)

shots_df["goal_weighted_centrality_end"] = 1 - np.abs(dy_end) / Y_CENTER

# ------------------------------
# 4. Linear regression (xG)
# ------------------------------
features = [
    "dist_to_goal_start",
    "dist_to_goal_start_sq",
    "goal_angle_start",
    "goal_weighted_centrality_start",
    "dist_to_goal_end",
    "dist_to_goal_end_sq",
    "goal_angle_end",
    "goal_weighted_centrality_end",
]

X = shots_df[features]
y = shots_df["shotXG"]

# Add constant for statsmodels
X_sm = sm.add_constant(X)

linreg_sm = sm.OLS(y, X_sm).fit()

print("\n================ LINEAR REGRESSION SUMMARY ================\n")
print(linreg_sm.summary())

# ------------------------------
# 5. Predict model xG
# ------------------------------
shots_df["model_xG"] = linreg_sm.predict(X_sm)

# Keep highest xG shot per pass
shots_df_max = shots_df.loc[
    shots_df.groupby("passEventId")["model_xG"].idxmax()
].copy()

# ------------------------------
# 6. Save outputs
# ------------------------------
shots_df_max[
    ["matchId", "passEventId", "model_xG"]
].to_csv(
    "shots_with_goal_probability_linreg.csv",
    index=False
)

joblib.dump(linreg_sm, "shot_xg_linear_model.joblib")

print("\nSaved shots_with_goal_probability_linreg.csv")
print("Saved shot_xg_linear_model.joblib")
