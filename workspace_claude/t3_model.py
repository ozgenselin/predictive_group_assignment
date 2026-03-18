"""
t3_model.pycinda activatw
-----------
Logistic Regression classifier on the UCI Adult Income dataset.

Pipeline:
  1. Load CSV, drop rows with '?' sentinel missing values.
  2. Separate features from target; encode target as binary int.
  3. Split into 80/20 stratified train/test (random_state=42).
  4. Build a ColumnTransformer pipeline:
       - Numerical features  → StandardScaler
       - Categorical features → OneHotEncoder (handle_unknown='ignore')
  5. Train LogisticRegression(max_iter=1000, random_state=42).
  6. Evaluate on test set: accuracy, precision, recall, F1, confusion matrix.
"""

import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── 1. Load & preprocess ──────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adult_income.csv")

df = pd.read_csv(DATA_PATH)

# Strip leading/trailing whitespace from all string columns
df = df.apply(lambda col: col.str.strip() if col.dtype == object else col)

# Drop rows that contain the '?' sentinel in any column
df = df[~(df == "?").any(axis=1)].reset_index(drop=True)

# ── 2. Feature / target split ─────────────────────────────────────────────────

TARGET = "income"
X = df.drop(columns=[TARGET])
# Encode target: '>50K' → 1, '<=50K' → 0
y = (df[TARGET] == ">50K").astype(int)

NUMERICAL_FEATURES = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# ── 3. Stratified 80/20 split ─────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# ── 4. Preprocessing + model pipeline ────────────────────────────────────────

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERICAL_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)

# ── 5. Train ──────────────────────────────────────────────────────────────────

model.fit(X_train, y_train)

# ── 6. Evaluate ───────────────────────────────────────────────────────────────

y_pred = model.predict(X_test)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
cm        = confusion_matrix(y_test, y_pred)

print("=" * 50)
print("  Logistic Regression — Test Set Evaluation")
print("=" * 50)
print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print()
print("  Confusion Matrix (rows=actual, cols=predicted):")
print(f"            Pred <=50K  Pred >50K")
print(f"  Act <=50K   {cm[0,0]:>6}     {cm[0,1]:>6}")
print(f"  Act  >50K   {cm[1,0]:>6}     {cm[1,1]:>6}")
print()
print("  Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=["<=50K", ">50K"]))
print("=" * 50)

# ── 7. Dataset / split sizes (for verification) ───────────────────────────────

print(f"\n  [Info] Clean rows total : {len(df)}")
print(f"  [Info] Training samples : {len(X_train)}")
print(f"  [Info] Test samples     : {len(X_test)}")
print(f"  [Info] Class balance (test) — <=50K: "
      f"{(y_test==0).sum()} ({(y_test==0).mean():.1%})  "
      f">50K: {(y_test==1).sum()} ({(y_test==1).mean():.1%})")
