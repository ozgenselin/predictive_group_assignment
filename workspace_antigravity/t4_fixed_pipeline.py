# t4_fixed_pipeline.py
# UCI Adult Income Dataset — Logistic Regression Baseline (Fixed)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv('adult_income.csv')

# BUG 1 FIX: Strip whitespace before checking for '?' missing values.
# Original code: df.replace('?', np.nan, inplace=True)
# Statistical Consequence: The original code failed to match ' ?' (with leading whitespace), meaning the '?' missing values 
# were NOT dropped. Instead, they were retained and treated as a distinct, valid category. This introduces noise and 
# bias by forcing the model to learn weights for missingness itself. Stripping whitespace correctly catches and drops these 3620 invalid rows.
df_obj = df.select_dtypes(['object'])
df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# ── Encode target ──────────────────────────────────────────────────────────────
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

# BUG 2 FIX: One-Hot Encode all nominal categorical features instead of Label/Ordinal Encoding.
# Original code: Mapped 'workclass' to arbitrary integers (0-7) and used LabelEncoder for other categorical columns.
# Statistical Consequence: Logistic Regression is a linear model. Assigning integer labels to nominal categories (e.g., Local-gov=4, Private=3) 
# artificially imposes an ordinal magnitude and mathematical distance between them (implying 4 > 3). This severely distorts the learned weights. 
# Using pd.get_dummies() creates independent indicator (boolean) features for each class, allowing the model to learn a separate, unbiased coefficient for each category.
cat_cols = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country'
]

# ── Separate features and target ───────────────────────────────────────────────
X = df.drop('income', axis=1)
y = df['income']

# Apply One-Hot Encoding
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# ── Train / test split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_cols = [
    'age', 'fnlwgt', 'education-num',
    'capital-gain', 'capital-loss', 'hours-per-week'
]

# BUG 3 FIX: Fit StandardScaler only on the training set to prevent data leakage.
# Original code: X[numeric_cols] = scaler.fit_transform(X[numeric_cols]) before the train_test_split.
# Statistical Consequence: Fitting the scaler on the entire dataset leaks information about the test set's distribution 
# (specifically, its mean and variance) into the training process. This violates strict test isolation, resulting in 
# artificially inflated, optimistic evaluation metrics. Fitting only on X_train ensures realistic generalization estimates.
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[numeric_cols])
X_test_num = scaler.transform(X_test[numeric_cols])

X_train.loc[:, numeric_cols] = X_train_num
X_test.loc[:, numeric_cols] = X_test_num

# ── Train model ────────────────────────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, random_state=42)
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
