import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ------------------------------
# Load dataset
# ------------------------------
df = pd.read_csv("passes_with_goal_features_and_prob.csv")

# ------------------------------
# Train logistic regression
# ------------------------------
features = ["log_dist_to_goal", "goal_angle"]
X = df[features]
y = df["becameShot"].astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# ------------------------------
# Create pitch grid
# ------------------------------
x_vals = np.linspace(0, 100, 100)
y_vals = np.linspace(0, 100, 100)
xx, yy = np.meshgrid(x_vals, y_vals)

# ------------------------------
# Calculate features on the grid
# ------------------------------
x_goal = 100
y_left, y_right = 37, 63  # Inner goalposts

dist_to_goal = np.sqrt((x_goal - xx)**2 + (50 - yy)**2)
log_dist_to_goal = np.log1p(dist_to_goal)

angle_left  = np.arctan2(y_left - yy, x_goal - xx)
angle_right = np.arctan2(y_right - yy, x_goal - xx)
goal_angle = np.abs(angle_right - angle_left)

grid_features = np.column_stack((log_dist_to_goal.ravel(), goal_angle.ravel()))
grid_features_scaled = scaler.transform(grid_features)

# Predict probabilities
probs = model.predict_proba(grid_features_scaled)[:,1]
probs_grid = probs.reshape(xx.shape)

# ------------------------------
# Plot pitch
# ------------------------------
fig, ax = plt.subplots(figsize=(12,6))

# Heatmap
heatmap = ax.imshow(probs_grid, origin='lower', extent=[0,100,0,100],
                    cmap='Reds', alpha=1)

# Pitch outline
ax.plot([0,100,100,0,0],[0,0,100,100,0], color='black', linewidth=2)

# Halfway line
ax.plot([50,50],[0,100], color='black', linewidth=1, linestyle='--')

# Goal (inner)
inner_goal = patches.Rectangle((100-0.5,37), 0.5, 26, linewidth=2, edgecolor='blue', facecolor='blue')
ax.add_patch(inner_goal)
ax.text(101,50,"Goal", va='center', color='white', fontsize=12, fontweight='bold')

# Outer goalposts
# Outer goalposts (lighter, dashed)
outer_goal = patches.Rectangle(
    (94,37),       # position
    6,             # width
    26,            # height
    linewidth=2,
    edgecolor='cyan',
    alpha=0.2,
    facecolor='none',
    linestyle='--'      # dashed line
)
ax.add_patch(outer_goal)

ax.add_patch(outer_goal)

# Goal box / penalty area (solid lines)
# Left vertical
ax.plot([84,84],[19,81], color='green', linewidth=2)
# Right vertical (goal line)
ax.plot([100,100],[19,81], color='green', linewidth=2)
# Top horizontal
ax.plot([84,100],[81,81], color='green', linewidth=2)
# Bottom horizontal
ax.plot([84,100],[19,19], color='green', linewidth=2)

# Penalty arc (semi-circle outside the penalty area, facing pitch)
penalty_arc = patches.Arc(
    (84, 50),   # center along penalty spot line, middle of pitch
    width=18,   # diameter along X
    height=18,  # diameter along Y
    angle=0,    # no rotation
    theta1=90,  # start at top
    theta2=270, # end at bottom
    color='green',
    linewidth=2
)
ax.add_patch(penalty_arc)




# Center circle
center_circle = plt.Circle((50,50), 10, color='black', fill=False, linewidth=1)
ax.add_patch(center_circle)

# Labels
ax.set_xlabel("X (pitch length)")
ax.set_ylabel("Y (pitch width)")
ax.set_title("Shot Probability by Pass End Location (Wyscout Coordinates)")
fig.colorbar(heatmap, ax=ax, label="Probability of shot within 15s")

plt.show()
