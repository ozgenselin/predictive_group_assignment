# t4_fixed_pipeline.py
# UCI Adult Income Dataset — Logistic Regression Baseline (Fixed)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
# BUG REMOVED: LabelEncoder import dropped — LabelEncoder was used to apply
# arbitrary ordinal integer codes to nominal categorical columns, which is
# statistically incorrect for logistic regression (see fix below).
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

# ── Workclass encoding ─────────────────────────────────────────────────────────
# BUG: The original code mapped 'workclass' to arbitrary integers
# (e.g. Private→3, Federal-gov→0, Never-worked→7). 'workclass' is a *nominal*
# categorical variable — its categories have no natural numeric ordering.
# Logistic regression treats these integers as a single continuous feature and
# fits one coefficient to the entire column, which means it assumes a linear
# relationship: "Never-worked (7)" is modelled as exactly 7× "Federal-gov (0)".
# This introduces systematic bias in the workclass coefficient.
#
# FIX: One-hot encode 'workclass' with pd.get_dummies (drop_first=True avoids
# perfect multicollinearity / the dummy-variable trap). Each category receives
# its own binary indicator column, allowing logistic regression to estimate an
# independent coefficient per category — the statistically correct representation
# for a nominal variable.
df['workclass'] = df['workclass'].str.strip()
df = pd.get_dummies(df, columns=['workclass'], drop_first=True)

# ── Encode remaining categorical columns ──────────────────────────────────────
cat_cols = [
    'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country'
]
# BUG: LabelEncoder assigned arbitrary integer codes to each nominal category
# (e.g. in 'education': Bachelors→1, Masters→5, Doctorate→3, etc.).
# Because the codes are arbitrary, their numeric differences are meaningless,
# yet logistic regression treats the column as a continuous ordered variable
# and fits a single slope coefficient to it. This distorts all categorical
# coefficients and can push predictions in the wrong direction.
#
# FIX: One-hot encode all remaining nominal categoricals with pd.get_dummies.
# drop_first=True drops one level per variable to avoid multicollinearity.
# Each category now gets its own binary feature and its own model coefficient,
# removing the false ordinal assumption entirely.
for col in cat_cols:
    df[col] = df[col].astype(str).str.strip()
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# ── Separate features and target ───────────────────────────────────────────────
X = df.drop('income', axis=1)
y = df['income']

numeric_cols = [
    'age', 'fnlwgt', 'education-num',
    'capital-gain', 'capital-loss', 'hours-per-week'
]

# ── Train / test split ─────────────────────────────────────────────────────────
# BUG FIX: The split is moved to BEFORE the scaler is fitted (see scaler section
# below). The scaler must never see test-set rows during fit; placing the split
# here ensures that only X_train is passed to scaler.fit_transform().
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Fitting StandardScaler ─────────────────────────────────────────────────────
# BUG: The original code called scaler.fit_transform(X[numeric_cols]) on the
# *entire* dataset before the train/test split. The scaler computes mean and
# standard deviation across all rows — including the test rows. Those test-set
# statistics then influence how every training sample is normalised, constituting
# data leakage. Evaluation metrics become optimistically biased because the
# preprocessing step has effectively "seen" the test set.
#
# FIX: Call fit_transform() only on X_train (training data only), then call
# transform() — *not* fit_transform() — on X_test. The test set is normalised
# using training-set statistics alone, preserving full independence of the
# holdout evaluation and producing unbiased performance estimates.
scaler = StandardScaler()
X_train = X_train.copy()   # avoid pandas SettingWithCopyWarning on slice
X_test  = X_test.copy()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])  # fit on train only
X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])       # transform only — no fit

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
