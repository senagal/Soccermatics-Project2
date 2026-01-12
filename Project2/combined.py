# -*- coding: utf-8 -*-
"""
Full pass-to-goal probability pipeline using all_passes_with_future_shots_detailed.csv
Includes model summaries, ROC-AUC, and computes danger pass threshold.
"""

import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, roc_curve

# ------------------------------
# 1️⃣ Load all passes
# ------------------------------
df = pd.read_csv("all_passes_with_future_shots_detailed.csv")

# Keep only the necessary columns
df = df[
    [
        "matchId", "passEventId", "passTeamId",
        "passPlayerId", "passPlayerName", "passPlayerPosition",
        "passTimeSeconds", "startX", "startY", "endX", "endY", "becameShot", "minutesPlayed"
    ]
]

# ------------------------------
# 2️⃣ Load pre-trained logistic regression model
# ------------------------------
logreg = joblib.load("logreg_pass_to_shot_model.joblib")
scaler = joblib.load("logreg_pass_to_shot_scaler.joblib")

# ------------------------------
# 3️⃣ Prepare features for logistic regression
# ------------------------------
X_GOAL = 100
Y_CENTER = 50
Y_LEFT, Y_RIGHT = 37, 63

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

df["goal_weighted_centrality_start"] = (1 - np.abs(df["startY"] - Y_CENTER)/Y_CENTER) * (df["startX"]/X_GOAL)
df["goal_weighted_centrality_end"] = (1 - np.abs(df["endY"] - Y_CENTER)/Y_CENTER) * (df["endX"]/X_GOAL)
df["pass_direction"] = np.arctan2(df["endY"] - df["startY"], df["endX"] - df["startX"])

features_logreg = [
    "log_dist_to_goal_start", "goal_angle_start", "goal_weighted_centrality_start",
    "log_dist_to_goal_end", "goal_angle_end", "goal_weighted_centrality_end",
    "pass_direction"
]

X = df[features_logreg]
df["shot_probability"] = logreg.predict_proba(scaler.transform(X))[:, 1]

# ------------------------------
# Logistic regression model summary & ROC-AUC
# ------------------------------
y_true = df["becameShot"]
y_pred = df["shot_probability"]

roc_auc = roc_auc_score(y_true, y_pred)
fpr, tpr, thresholds = roc_curve(y_true, y_pred)

print(f"Logistic Regression (Shot) ROC-AUC: {roc_auc:.4f}")
print("ROC curve thresholds sample:", thresholds[:5], "...")

# ------------------------------
# 4️⃣ Load pre-trained linear regression model for xG
# ------------------------------
linreg_sm = sm.load("shot_xg_linear_model_sm.pkl")

# ------------------------------
# 5️⃣ Prepare features for linear regression
# ------------------------------
dxs = X_GOAL - df["startX"]
dys = df["startY"] - Y_CENTER
dxe = X_GOAL - df["endX"]
dye = df["endY"] - Y_CENTER

df_lin = pd.DataFrame({
    "dist_to_goal_start": np.sqrt(dxs**2 + dys**2),
})
df_lin["dist_to_goal_start_sq"] = df_lin["dist_to_goal_start"]**2
df_lin["goal_angle_start"] = np.abs(
    np.arctan2(Y_LEFT - df["startY"], X_GOAL - df["startX"]) - 
    np.arctan2(Y_RIGHT - df["startY"], X_GOAL - df["startX"])
)
df_lin["goal_weighted_centrality_start"] = 1 - np.abs(dys)/Y_CENTER

df_lin["dist_to_goal_end"] = np.sqrt(dxe**2 + dye**2)
df_lin["dist_to_goal_end_sq"] = df_lin["dist_to_goal_end"]**2
df_lin["goal_angle_end"] = np.abs(
    np.arctan2(Y_LEFT - df["endY"], X_GOAL - df["endX"]) -
    np.arctan2(Y_RIGHT - df["endY"], X_GOAL - df["endX"])
)
df_lin["goal_weighted_centrality_end"] = 1 - np.abs(dye)/Y_CENTER

X_lin_sm = sm.add_constant(df_lin)
df["max_model_xG"] = linreg_sm.predict(X_lin_sm)

# Linear regression summary
print("\nLinear Regression (xG) Summary:")
print(linreg_sm.summary())

# ------------------------------
# 6️⃣ Compute pass_goal_probability
# ------------------------------
df["pass_goal_probability"] = df["shot_probability"] * df["max_model_xG"]

# ------------------------------
# 7️⃣ Compute danger pass threshold
# ------------------------------
threshold = df["pass_goal_probability"].quantile(0.90)  # top 10%
print(f"\nDanger pass threshold (top 10%): {threshold:.4f}")

df["is_danger_pass"] = df["pass_goal_probability"] >= threshold

# ------------------------------
# 8️⃣ Save output
# ------------------------------
output_cols = [
    "matchId","passEventId","passTeamId","passPlayerId","passPlayerName","passPlayerPosition",
    "minutesPlayed","passTimeSeconds","startX","startY","endX","endY",
    "becameShot","max_model_xG","shot_probability","pass_goal_probability","is_danger_pass"
]

df = df[output_cols]
df.to_csv("passes_with_goal_probabilities.csv", index=False)
print("\nSaved passes_with_goal_probabilities.csv with all passes, shot/xG models, and danger pass flag.")
