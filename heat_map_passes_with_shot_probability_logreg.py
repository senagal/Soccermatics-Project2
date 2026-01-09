# -*- coding: utf-8 -*-
"""
Created on Fri Jan 9 16:16:52 2026

@author: senaa
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------------- Load model ----------------
model = joblib.load("logreg_pass_to_shot_model.joblib")
scaler = joblib.load("logreg_pass_to_shot_scaler.joblib")

# ---------------- Constants ----------------
X_GOAL = 100
Y_CENTER = 50
Y_LEFT, Y_RIGHT = 37, 63

# ---------------- Feature engineering ----------------
def compute_features(df):
    df = df.copy()

    df["dist_to_goal_end"] = np.sqrt(
        (X_GOAL - df["endX"])**2 +
        (Y_CENTER - df["endY"])**2
    )
    df["log_dist_to_goal_end"] = np.log1p(df["dist_to_goal_end"])

    angle_left = np.arctan2(Y_LEFT - df["endY"], X_GOAL - df["endX"])
    angle_right = np.arctan2(Y_RIGHT - df["endY"], X_GOAL - df["endX"])
    df["goal_angle_end"] = np.abs(angle_right - angle_left)

    df["goal_weighted_centrality_end"] = (
        (1 - np.abs(df["endY"] - Y_CENTER) / Y_CENTER) *
        (df["endX"] / X_GOAL)
    )

    return df

# ---------------- Pitch plotting ----------------
def plot_pitch_heatmap(heatmap, title, xlabel, ylabel, cmap="Reds", xlim=(40,100)):
    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(
        heatmap,
        origin="lower",
        extent=[40, 100, 0, 100],  # Only show 40 → 100
        cmap=cmap,
        vmin=0,
        vmax=1
    )

    fig.colorbar(im, ax=ax, label="P(Shot within 15s)")

    # Pitch outline (adjusted to 40 → 100 for visible area)
    ax.plot([40,100,100,40,40],[0,0,100,100,0], color="black", linewidth=2)
    ax.plot([50,50],[0,100], linestyle="--", color="black")  # halfway-ish line for visible area

    # Penalty area
    ax.plot([84,84],[19,81], color="green", linewidth=2)
    ax.plot([100,100],[19,81], color="green", linewidth=2)
    ax.plot([84,100],[81,81], color="green", linewidth=2)
    ax.plot([84,100],[19,19], color="green", linewidth=2)
    ax.add_patch(patches.Arc((84,50), 18, 18, theta1=90, theta2=270,
                             color="green", linewidth=2))

    # Center circle (partial visible)
    ax.add_patch(plt.Circle((50,50), 10, fill=False))

    # Inner goal line
    ax.plot([100,94,94,100,100], [37,37,63,63,37], color="blue", alpha=0.6, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

# ---------------- Grid ----------------
x_vals = np.linspace(40, 100, 120)  # only 40-100
y_vals = np.linspace(0, 100, 100)
xx, yy = np.meshgrid(x_vals, y_vals)

FEATURES = [
    "log_dist_to_goal_end",
    "goal_angle_end",
    "goal_weighted_centrality_end",
    "pass_direction"
]

# =====================================================
# HEATMAP 1 — PASS START LOCATION
# =====================================================
grid_start = pd.DataFrame({
    "startX": xx.ravel(),
    "startY": yy.ravel(),
    "endX": 70,   # end somewhere realistic
    "endY": 50
})

grid_start["pass_direction"] = np.arctan2(
    grid_start["endY"] - grid_start["startY"],
    grid_start["endX"] - grid_start["startX"]
)

grid_start = compute_features(grid_start)
X_scaled = scaler.transform(grid_start[FEATURES])
grid_start["shot_probability"] = model.predict_proba(X_scaled)[:, 1]

heatmap_start = grid_start["shot_probability"].values.reshape(len(y_vals), len(x_vals))

plot_pitch_heatmap(
    heatmap_start,
    "Shot Probability by PASS START Location (x >= 40)",
    "Start X",
    "Start Y"
)

# =====================================================
# HEATMAP 2 — PASS END LOCATION
# =====================================================
grid_end = pd.DataFrame({
    "endX": xx.ravel(),
    "endY": yy.ravel()
})

grid_end["startX"] = grid_end["endX"] - 10
grid_end["startY"] = grid_end["endY"]
grid_end["pass_direction"] = 0.0

grid_end = compute_features(grid_end)
X_scaled = scaler.transform(grid_end[FEATURES])
grid_end["shot_probability"] = model.predict_proba(X_scaled)[:, 1]

heatmap_end = grid_end["shot_probability"].values.reshape(len(y_vals), len(x_vals))

plot_pitch_heatmap(
    heatmap_end,
    "Shot Probability by PASS END Location (x >= 40)",
    "End X",
    "End Y"
)
