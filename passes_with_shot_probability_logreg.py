# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 16:01:34 2026

@author: senaa
"""
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# ------------------------------
# 1. Load dataset
# ------------------------------
df = pd.read_csv("all_passes_with_future_shots_detailed.csv")

# ------------------------------
# 2. Constants (Wyscout pitch)
# ------------------------------
X_GOAL = 100
Y_CENTER = 50
Y_LEFT, Y_RIGHT = 37, 63

# ------------------------------
# 3. Feature engineering
# ------------------------------

# --- Distance to goal (end location) ---
df["dist_to_goal_end"] = np.sqrt(
    (X_GOAL - df["endX"]) ** 2 +
    (Y_CENTER - df["endY"]) ** 2
)
df["log_dist_to_goal_end"] = np.log1p(df["dist_to_goal_end"])

# --- Goal angle (end location) ---
angle_left = np.arctan2(Y_LEFT - df["endY"], X_GOAL - df["endX"])
angle_right = np.arctan2(Y_RIGHT - df["endY"], X_GOAL - df["endX"])
df["goal_angle_end"] = np.abs(angle_right - angle_left)

# --- Goal-weighted centrality (end location) ---
df["goal_weighted_centrality_end"] = (
    (1 - np.abs(df["endY"] - Y_CENTER) / Y_CENTER) *
    (df["endX"] / X_GOAL)
)

# --- Pass direction (start → end) ---
df["pass_direction"] = np.arctan2(
    df["endY"] - df["startY"],
    df["endX"] - df["startX"]
)

# ------------------------------
# 4. Features & target
# ------------------------------
features = [
    "log_dist_to_goal_end",
    "goal_angle_end",
    "goal_weighted_centrality_end",
    "pass_direction",
]

X = df[features]
y = df["becameShot"].astype(int)

# ------------------------------
# 5. Train-test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ------------------------------
# 6. Scale features
# ------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# 7. Fit logistic regression
# ------------------------------
model = LogisticRegression(
    penalty="l2",
    C=0.5,
    max_iter=2000,
    solver="lbfgs"
)

model.fit(X_train_scaled, y_train)

# ------------------------------
# 8. Evaluation
# ------------------------------
y_pred = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

roc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC-AUC: {roc:.3f}\n")

# ------------------------------
# 9. Coefficient table
# ------------------------------
coef_df = pd.DataFrame({
    "feature": features,
    "coefficient": model.coef_[0]
}).sort_values("coefficient", ascending=False)

print("Model coefficients:\n")
print(coef_df)

# ------------------------------
# 10. Predict for ALL passes
# ------------------------------
X_all_scaled = scaler.transform(X)
df["shot_probability"] = model.predict_proba(X_all_scaled)[:, 1]

# ------------------------------
# 11. Save model & scaler
# ------------------------------
joblib.dump(model, "logreg_pass_to_shot_model.joblib")
joblib.dump(scaler, "logreg_pass_to_shot_scaler.joblib")

print("\nSaved model to logreg_pass_to_shot_model.joblib")
print("Saved scaler to logreg_pass_to_shot_scaler.joblib")

# ------------------------------
# 12. Save dataset with predictions
# ------------------------------
df[
    [
        "matchId",
        "passEventId",
        "becameShot",
        "endX",
        "endY",
        "log_dist_to_goal_end",
        "goal_angle_end",
        "goal_weighted_centrality_end",
        "pass_direction",
        "shot_probability",
    ]
].to_csv(
    "passes_with_shot_probability_logreg.csv",
    index=False
)

print("\nSaved passes_with_shot_probability_logreg.csv")
