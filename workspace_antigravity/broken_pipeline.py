# broken_pipeline.py
# UCI Adult Income Dataset — Logistic Regression Baseline

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv('adult_income.csv')

# Replace non-standard missing value marker and drop incomplete rows
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# ── Encode target ──────────────────────────────────────────────────────────────
df['income'] = df['income'].str.strip().map({'<=50K': 0, '>50K': 1})

# ── Workclass ordinal encoding ─────────────────────────────────────────────────
workclass_map = {
    'Private':          3,
    'Self-emp-not-inc': 1,
    'Self-emp-inc':     2,
    'Federal-gov':      0,
    'Local-gov':        4,
    'State-gov':        5,
    'Without-pay':      6,
    'Never-worked':     7
}
df['workclass'] = df['workclass'].str.strip().map(workclass_map)

# ── Encode remaining categorical columns ──────────────────────────────────────
cat_cols = [
    'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country'
]
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str).str.strip())

# ── Separate features and target ───────────────────────────────────────────────
X = df.drop('income', axis=1)
y = df['income']

numeric_cols = [
    'age', 'fnlwgt', 'education-num',
    'capital-gain', 'capital-loss', 'hours-per-week'
]

# ── Fitting StandardScaler ─────────────────────────────────────────────────────
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# ── Train / test split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Train model ────────────────────────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

print("=" * 50)
print("  Model Evaluation — broken_pipeline.py")
print("=" * 50)
print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"  F1-Score  : {f1_score(y_test, y_pred):.4f}")
print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
print()
print("  Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()
print("  Classification Report:")
print(classification_report(y_test, y_pred))
