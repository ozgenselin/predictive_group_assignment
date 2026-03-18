# t4_fixed_pipeline.py
# UCI Adult Income Dataset — Logistic Regression Baseline

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv('adult_income.csv')

# Replace non-standard missing value marker and drop incomplete rows
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# ── Encode target ──────────────────────────────────────────────────────────────
df['income'] = df['income'].str.strip().map({'<=50K': 0, '>50K': 1})

# ── Separate features and target ───────────────────────────────────────────────
X = df.drop('income', axis=1)
y = df['income']

numeric_cols = [
    'age', 'fnlwgt', 'education-num',
    'capital-gain', 'capital-loss', 'hours-per-week'
]
categorical_cols = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country'
]

# ── Train / test split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Correction: treat all nominal categories, including workclass, with one-hot encoding instead of arbitrary
# integer codes because logistic regression would otherwise interpret fake order and distance as real signal.
preprocessor = ColumnTransformer(
    transformers=[
        # Correction: fit scaling only on the training fold inside a pipeline because fitting StandardScaler
        # before the split leaks test-set moments into training and inflates evaluation validity.
        ('num', StandardScaler(), numeric_cols),
        # Correction: fit category encoding only on the training fold because learning category structure from
        # the full dataset leaks information across the split and biases the held-out evaluation.
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ]
)

# ── Train model ────────────────────────────────────────────────────────────────
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])
model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

print("=" * 50)
print("  Model Evaluation — t4_fixed_pipeline.py")
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
