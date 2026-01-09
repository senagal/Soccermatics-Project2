# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 16:08:03 2026

@author: senaa
"""

# -*- coding: utf-8 -*-
"""
Model evaluation & summary for pass → shot logistic regression
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score
)

# ------------------------------
# 1. Load model & scaler
# ------------------------------
model = joblib.load("logreg_pass_to_shot_model.joblib")
scaler = joblib.load("logreg_pass_to_shot_scaler.joblib")

# ------------------------------
# 2. Load dataset
# ------------------------------
df = pd.read_csv("all_passes_with_future_shots_detailed.csv")

# ------------------------------
# 3. Recreate features (MUST MATCH TRAINING)
# ------------------------------
X_GOAL = 100
Y_CENTER = 50
Y_LEFT, Y_RIGHT = 37, 63

# Distance to goal (end)
df["dist_to_goal_end"] = np.sqrt(
    (X_GOAL - df["endX"]) ** 2 +
    (Y_CENTER - df["endY"]) ** 2
)
df["log_dist_to_goal_end"] = np.log1p(df["dist_to_goal_end"])

# Goal angle (end)
angle_left = np.arctan2(Y_LEFT - df["endY"], X_GOAL - df["endX"])
angle_right = np.arctan2(Y_RIGHT - df["endY"], X_GOAL - df["endX"])
df["goal_angle_end"] = np.abs(angle_right - angle_left)

# Goal-weighted centrality (end)
df["goal_weighted_centrality_end"] = (
    (1 - np.abs(df["endY"] - Y_CENTER) / Y_CENTER) *
    (df["endX"] / X_GOAL)
)

# Pass direction
df["pass_direction"] = np.arctan2(
    df["endY"] - df["startY"],
    df["endX"] - df["startX"]
)

features = [
    "log_dist_to_goal_end",
    "goal_angle_end",
    "goal_weighted_centrality_end",
    "pass_direction",
]

X = df[features]
y_true = df["becameShot"].astype(int)

# ------------------------------
# 4. Scale + predict
# ------------------------------
X_scaled = scaler.transform(X)

y_pred = model.predict(X_scaled)
y_pred_prob = model.predict_proba(X_scaled)[:, 1]

# ------------------------------
# 5. Overall metrics
# ------------------------------
acc = accuracy_score(y_true, y_pred)
roc = roc_auc_score(y_true, y_pred_prob)

print("\n=== OVERALL PERFORMANCE ===")
print(f"Accuracy: {acc:.3f}")
print(f"ROC-AUC : {roc:.3f}")

print("\nClassification report:")
print(classification_report(y_true, y_pred, digits=3))

# ------------------------------
# 6. Confusion matrix (percentages)
# ------------------------------
cm = confusion_matrix(y_true, y_pred)

tn, fp, fn, tp = cm.ravel()

non_shot_total = tn + fp
shot_total = tp + fn

non_shot_correct_pct = tn / non_shot_total * 100
shot_correct_pct = tp / shot_total * 100

print("\n=== CLASS-WISE ACCURACY ===")
print(f"Non-shot passes correctly predicted: {non_shot_correct_pct:.2f}%")
print(f"Shot passes correctly predicted    : {shot_correct_pct:.2f}%")

# ------------------------------
# 7. Confusion matrix (pretty)
# ------------------------------
cm_df = pd.DataFrame(
    cm,
    index=["Actual Non-Shot", "Actual Shot"],
    columns=["Pred Non-Shot", "Pred Shot"]
)

print("\nConfusion Matrix (counts):")
print(cm_df)

cm_pct = cm_df.div(cm_df.sum(axis=1), axis=0) * 100

print("\nConfusion Matrix (row %):")
print(cm_pct.round(2))
