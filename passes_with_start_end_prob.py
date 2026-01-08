# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 11:39:36 2026

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
df = pd.read_csv("all_passes_with_becameShot.csv")  # must have startX, startY, endX, endY, becameShot

# ------------------------------
# 2. Define features and target
# ------------------------------
# Using both start and end coordinates
features = ["startX", "startY", "endX", "endY"]
X = df[features]
y = df["becameShot"].astype(int)

# ------------------------------
# 3. Train-test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ------------------------------
# 4. Scale features
# ------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# 5. Fit logistic regression
# ------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# ------------------------------
# 6. Evaluate model
# ------------------------------
y_pred_prob = model.predict_proba(X_test_scaled)[:,1]
y_pred = model.predict(X_test_scaled)

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

roc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC-AUC: {roc:.3f}\n")

# ------------------------------
# 7. Interpret coefficients
# ------------------------------
coefficients = pd.DataFrame({
    "feature": features,
    "coef": model.coef_[0]
})
print("Model coefficients:\n", coefficients)

# ------------------------------
# 8. Predict shot probability for ALL passes
# ------------------------------
X_scaled_all = scaler.transform(X)
df["shot_probability"] = model.predict_proba(X_scaled_all)[:,1]

# ------------------------------
# 9. Save dataset with features and predicted probability
# ------------------------------
df[["matchId", "passEventId", "startX", "startY", "endX", "endY", "becameShot", "shot_probability"]].to_csv(
    "passes_with_start_end_prob.csv", index=False
)
print("\nSaved passes_with_start_end_prob.csv")
