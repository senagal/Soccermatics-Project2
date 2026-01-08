# -*- coding: utf-8 -*- 
"""
Created on Wed Jan  8 2026

@author: senaa
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# ------------------------------
# 1. Load passes dataset
# ------------------------------
df = pd.read_csv("all_passes_with_becameShot.csv")  # must have endX, endY, becameShot

# ------------------------------
# 2. Calculate distance to goal (log) and goal angle
# ------------------------------

# Goalposts coordinates
x_goal = 100
y_left, y_right = 37, 63  # Wyscout coordinates

# Distance to goal (Euclidean) and log-transform
df["dist_to_goal"] = np.sqrt((x_goal - df["endX"])**2 + (50 - df["endY"])**2)
df["log_dist_to_goal"] = np.log1p(df["dist_to_goal"])

# Goal angle in radians
angle_left  = np.arctan2(y_left - df["endY"], x_goal - df["endX"])
angle_right = np.arctan2(y_right - df["endY"], x_goal - df["endX"])
df["goal_angle"] = np.abs(angle_right - angle_left)

# ------------------------------
# 3. Define features and target
# ------------------------------
features = ["log_dist_to_goal", "goal_angle"]
X = df[features]
y = df["becameShot"].astype(int)

# ------------------------------
# 4. Train-test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ------------------------------
# 5. Scale features
# ------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# 6. Fit logistic regression
# ------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# ------------------------------
# 7. Evaluate model
# ------------------------------
y_pred_prob = model.predict_proba(X_test_scaled)[:,1]
y_pred = model.predict(X_test_scaled)

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

roc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC-AUC: {roc:.3f}\n")

# ------------------------------
# 8. Interpret coefficients
# ------------------------------
coefficients = pd.DataFrame({
    "feature": features,
    "coef": model.coef_[0]
})
print("Model coefficients:\n", coefficients)

# ------------------------------
# 9. Predict shot probability for ALL passes
# ------------------------------
# Scale all data
X_scaled_all = scaler.transform(X)
df["shot_probability"] = model.predict_proba(X_scaled_all)[:,1]

# ------------------------------
# 10. Save dataset with features and predicted probability
# ------------------------------
df[["matchId", "passEventId", "becameShot", "log_dist_to_goal", "goal_angle", "shot_probability"]].to_csv(
    "passes_with_goal_features_and_prob.csv", index=False
)
print("\nSaved passes_with_goal_features_and_prob.csv")
