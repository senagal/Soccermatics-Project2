# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 2026

@author: senaa
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

# ------------------------------
# 1. Load CSV with pass end coordinates and shot probability
# ------------------------------
df = pd.read_csv("passes_with_coordinates_features_and_prob.csv")

# ------------------------------
# 2. Create a grid for plotting
# ------------------------------
x_vals = np.linspace(0, 100, 100)
y_vals = np.linspace(0, 100, 100)
xx, yy = np.meshgrid(x_vals, y_vals)

# ------------------------------
# 3. Map shot probabilities from CSV onto the grid
# ------------------------------
from scipy.interpolate import griddata

# Use endX, endY as coordinates, shot_probability as values
points = df[['endX','endY']].values
values = df['shot_probability'].values

# Interpolate onto the grid
probs_grid = griddata(points, values, (xx, yy), method='linear')

# Fill NaNs with 0 (or could use nearest)
probs_grid = np.nan_to_num(probs_grid, nan=0)

# ------------------------------
# 4. Plot pitch with heatmap
# ------------------------------
fig, ax = plt.subplots(figsize=(12,6))

# Heatmap: green = low, dark red = high
heatmap = ax.imshow(probs_grid, origin='lower', extent=[0,100,0,100],
                    cmap='Reds', alpha=1, vmin=0, vmax=1)

# Colorbar
cbar = fig.colorbar(heatmap, ax=ax, label="Probability of shot within 15s")

# ------------------------------
# 5. Draw pitch outlines
# ------------------------------
# Pitch rectangle
ax.plot([0,100,100,0,0],[0,0,100,100,0], color='black', linewidth=2)
# Halfway line
ax.plot([50,50],[0,100], color='black', linewidth=1, linestyle='--')

# Goal box / penalty area
ax.plot([84,84],[19,81], color='green', linewidth=2)
ax.plot([100,100],[19,81], color='green', linewidth=2)
ax.plot([84,100],[81,81], color='green', linewidth=2)
ax.plot([84,100],[19,19], color='green', linewidth=2)

# Penalty arc
penalty_arc = patches.Arc((84,50), width=18, height=18, angle=0,
                          theta1=90, theta2=270, color='green', linewidth=2)
ax.add_patch(penalty_arc)

# Center circle
center_circle = plt.Circle((50,50), 10, color='black', fill=False, linewidth=1)
ax.add_patch(center_circle)

# Inner goal (just the goal line)
inner_goal = patches.Rectangle((100-0.5,37), 0.5, 26,
                               linewidth=2, edgecolor='blue', alpha=0.4, facecolor='none')
ax.add_patch(inner_goal)


# Outer goal frame (6 units back, dashed)
ax.plot([94, 94], [37, 63], color='cyan', linestyle='--', linewidth=2, alpha=0.4)  # left post back
ax.plot([94, 100], [63, 63], color='cyan', linestyle='--', linewidth=2, alpha=0.4) # top bar
ax.plot([94, 100], [37, 37], color='cyan', linestyle='--', linewidth=2, alpha=0.4) # bottom bar

# Labels
ax.set_xlabel("X (pitch length)")
ax.set_ylabel("Y (pitch width)")
ax.set_title("Shot Probability Heatmap by Pass End Location")

plt.show()
