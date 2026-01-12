# -*- coding: utf-8 -*-
"""
Visualising spatial effects of passes on:
1) Shot probability
2) xG
3) Pass-goal probability

Two heatmaps per metric:
- Pass START location
- Pass END location

Same pitch, same grid, same scale

@author: senaa
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import statsmodels.api as sm

# ======================================================
# LOAD MODELS
# ======================================================
logreg = joblib.load("logreg_pass_to_shot_model.joblib")
scaler = joblib.load("logreg_pass_to_shot_scaler.joblib")
linreg = joblib.load("shot_xg_linear_model.joblib")

# ======================================================
# CONSTANTS
# ======================================================
X_GOAL = 100
Y_CENTER = 50
Y_LEFT, Y_RIGHT = 37, 63

# ======================================================
# FEATURE ENGINEERING (IDENTICAL TO PIPELINE)
# ======================================================
def compute_features(df):
    df = df.copy()

    df["dist_to_goal_start"] = np.sqrt((X_GOAL - df["startX"])**2 + (Y_CENTER - df["startY"])**2)
    df["dist_to_goal_end"] = np.sqrt((X_GOAL - df["endX"])**2 + (Y_CENTER - df["endY"])**2)

    df["log_dist_to_goal_start"] = np.log1p(df["dist_to_goal_start"])
    df["log_dist_to_goal_end"] = np.log1p(df["dist_to_goal_end"])

    df["goal_angle_start"] = np.abs(
        np.arctan2(Y_LEFT - df["startY"], X_GOAL - df["startX"]) -
        np.arctan2(Y_RIGHT - df["startY"], X_GOAL - df["startX"])
    )

    df["goal_angle_end"] = np.abs(
        np.arctan2(Y_LEFT - df["endY"], X_GOAL - df["endX"]) -
        np.arctan2(Y_RIGHT - df["endY"], X_GOAL - df["endX"])
    )

    df["goal_weighted_centrality_start"] = (
        (1 - np.abs(df["startY"] - Y_CENTER) / Y_CENTER) * (df["startX"] / X_GOAL)
    )

    df["goal_weighted_centrality_end"] = (
        (1 - np.abs(df["endY"] - Y_CENTER) / Y_CENTER) * (df["endX"] / X_GOAL)
    )

    df["pass_direction"] = np.arctan2(
        df["endY"] - df["startY"],
        df["endX"] - df["startX"]
    )

    return df

FEATURES = [
    "log_dist_to_goal_start",
    "goal_angle_start",
    "goal_weighted_centrality_start",
    "log_dist_to_goal_end",
    "goal_angle_end",
    "goal_weighted_centrality_end",
    "pass_direction"
]

# ======================================================
# PITCH DRAWING
# ======================================================
def draw_pitch(ax):
    ax.set_facecolor("white")

    ax.plot([40,100,100,40,40], [0,0,100,100,0], color="black", linewidth=2)

    ax.plot([84,84],[19,81], color="black", linewidth=1)
    ax.plot([100,100],[19,81], color="black", linewidth=1)
    ax.plot([84,100],[81,81], color="black", linewidth=1)
    ax.plot([84,100],[19,19], color="black", linewidth=1)

    ax.add_patch(patches.Arc((84,50), 18, 18, theta1=90, theta2=270, linewidth=1))
    ax.plot([100,94,94,100], [37,37,63,63], color="black", linewidth=2)

    ax.set_xlim(40, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_yticks([])

# ======================================================
# GRID
# ======================================================
x_vals = np.linspace(40, 100, 120)
y_vals = np.linspace(0, 100, 100)
xx, yy = np.meshgrid(x_vals, y_vals)

# ======================================================
# GENERIC HEATMAP FUNCTION
# ======================================================
def plot_heatmap(z, title, cmap, label):
    fig, ax = plt.subplots(figsize=(12,6))
    im = ax.imshow(
        z,
        origin="lower",
        extent=[40,100,0,100],
        cmap=cmap,
        vmin=0,
        vmax=z.max()
    )
    draw_pitch(ax)
    plt.colorbar(im, ax=ax, label=label)
    ax.set_title(title)
    plt.show()

# ======================================================
# LOOP HELPER
# ======================================================
def make_grid(mode):
    if mode == "start":
        return pd.DataFrame({
            "startX": xx.ravel(),
            "startY": yy.ravel(),
            "endX": 70,
            "endY": 50
        }), "by Pass START Location"
    else:
        return pd.DataFrame({
            "endX": xx.ravel(),
            "endY": yy.ravel(),
            "startX": 60,
            "startY": 50
        }), "by Pass END Location"

# ======================================================
# 1️⃣ SHOT PROBABILITY
# ======================================================
for mode in ["start", "end"]:

    grid, title_suffix = make_grid(mode)
    grid = compute_features(grid)

    X_scaled = scaler.transform(grid[FEATURES])
    grid["shot_probability"] = logreg.predict_proba(X_scaled)[:, 1]

    heatmap = grid["shot_probability"].values.reshape(len(y_vals), len(x_vals))
    plot_heatmap(
        heatmap,
        f"Shot Probability {title_suffix}",
        cmap="Blues",
        label="P(Shot within 15s)"
    )

# ======================================================
# 2️⃣ xG (FIXED)
# ======================================================
for mode in ["start", "end"]:

    grid, title_suffix = make_grid(mode)
    grid = compute_features(grid)

    dx = X_GOAL - grid["endX"]
    dy = grid["endY"] - Y_CENTER

    X_xg = sm.add_constant(
        pd.DataFrame({
            "dist_to_goal": np.sqrt(dx**2 + dy**2),
            "dist_sq": dx**2 + dy**2,
            "goal_angle": grid["goal_angle_end"]
        }),
        has_constant="add"
    )

    grid["xG"] = linreg.predict(X_xg)

    heatmap = grid["xG"].values.reshape(len(y_vals), len(x_vals))
    plot_heatmap(
        heatmap,
        f"xG {title_suffix}",
        cmap="Greens",
        label="xG"
    )

# ======================================================
# 3️⃣ PASS → GOAL PROBABILITY
# ======================================================
for mode in ["start", "end"]:

    grid, title_suffix = make_grid(mode)
    grid = compute_features(grid)

    X_scaled = scaler.transform(grid[FEATURES])
    grid["shot_probability"] = logreg.predict_proba(X_scaled)[:, 1]

    dx = X_GOAL - grid["endX"]
    dy = grid["endY"] - Y_CENTER

    X_xg = sm.add_constant(
        pd.DataFrame({
            "dist_to_goal": np.sqrt(dx**2 + dy**2),
            "dist_sq": dx**2 + dy**2,
            "goal_angle": grid["goal_angle_end"]
        }),
        has_constant="add"
    )

    grid["xG"] = linreg.predict(X_xg)
    grid["pass_goal_probability"] = grid["shot_probability"] * grid["xG"]

    heatmap = grid["pass_goal_probability"].values.reshape(len(y_vals), len(x_vals))
    plot_heatmap(
        heatmap,
        f"Expected Danger (Pass → Goal) {title_suffix}",
        cmap="Reds",
        label="P(Goal from pass)"
    )
