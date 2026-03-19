# Agent Activity Log

## Template
- **Date/Time**: [YYYY-MM-DDTHH:MM:SSZ]
- **Goal**: [Brief description of the action]
- **User Request**: [What the human asked]
- **Files Changed**: [List of touched files]
- **Summary of Edits**: [High-level summary]
- **Verification Performed**: [What checks were run]

---
## Entry 1
- **Date/Time**: 2026-03-17T00:00:00Z
- **Goal**: Schema validation and encoded missing-value detection for the UCI Adult Income dataset
- **User Request**: Write a standalone Python script (`t1_schema_validation.py`) that loads `adult_income.csv`, produces a schema report (column names, dtypes, NaN counts, unique value counts for categoricals), detects non-standard/encoded missing values beyond NaN/blank, and prints a summary of the encoding used, total affected rows, and affected columns.
- **Files Changed**: `workspace_claude/t1_schema_validation.py` (created)
- **Summary of Edits**:
  - Script loads the CSV via a path relative to its own location (no hardcoded absolute paths).
  - Schema report iterates all 15 columns, printing dtype, NaN count, and unique-value count for object-dtype columns.
  - Sentinel detection checks all object columns against a set of known encoded-missing patterns (`"?"`, `"N/A"`, `""`, etc.) using both exact and strip-whitespace matching.
  - Row-level impact computed with a union mask across all flagged columns.
  - Summary block reports encoding(s) found, total affected rows, and affected column names.
- **Verification Performed**:
  - Ran `python3 t1_schema_validation.py` from `workspace_claude/`.
  - Confirmed 48,842 rows × 15 columns loaded correctly.
  - Confirmed zero standard NaN values (pandas did not interpret `"?"` as NaN on load).
  - Confirmed `"?"` sentinel found in `workclass` (2,799), `occupation` (2,809), `native-country` (857).
  - Total affected rows = 3,620 (union of rows, not sum, since some rows have `"?"` in multiple columns).
  - No hardcoded absolute paths present in the script.

---

## Entry 2
- **Date/Time**: 2026-03-17T13:14:00Z
- **Goal**: Full EDA of UCI Adult Income dataset with visualisations and narrative summary
- **User Request**: Write a standalone EDA script (`t2_eda.py`) that (1) drops '?' rows, (2) plots distributions for all continuous features and frequency charts for all categoricals, (3) produces a correlation heatmap, (4) generates grouped categorical comparisons using proportions, and (5) prints a written narrative summary identifying key patterns, class balance, and model implications.
- **Files Changed**:
  - `workspace_claude/t2_eda.py` (created)
  - `workspace_claude/eda_plots/` (directory created, 17 PNG files written)
- **Summary of Edits**:
  - Step 1: Loaded CSV, stripped whitespace, dropped 3,620 rows containing '?' in workclass/occupation/native-country → 45,222 clean rows.
  - Step 2: Overlapping histogram grid (2×3) for 6 continuous features, split by income class; saved as `01_continuous_distributions.png`.
  - Step 3: Individual frequency bar charts for each of 8 categorical features + target variable bar chart with count and percentage annotations; 9 files (`02_freq_*.png`, `03_target_distribution.png`).
  - Step 4: Lower-triangle Pearson correlation heatmap for numerical features; saved as `04_correlation_heatmap.png`. Strongest correlation: hours-per-week ↔ education-num (r=+0.146).
  - Step 5: Stacked 100% proportion bar charts (sorted by >50K rate) for 6 key categorical features vs income; 6 files (`05_grouped_*.png`). Proportions used throughout — no raw counts in grouped plots.
  - Step 6: Printed narrative summary with class balance stats (75.2% ≤50K, 24.8% >50K, 3:1 ratio) and three key patterns: (1) education-num threshold (48.7% vs 13.5% >50K rate), (2) marital-status structure (Married-civ-spouse 45.4%), (3) capital-gain sparsity (8.4% non-zero, 62.7% >50K among those).
- **Verification Performed**:
  - Ran `python3 t2_eda.py` from `workspace_claude/`; zero errors, zero warnings.
  - Confirmed all 17 PNG files exist with non-zero sizes (27K–159K).
  - Confirmed clean row count (45,222) matches Task 1 finding (48,842 − 3,620 = 45,222).
  - Confirmed grouped plots use `value_counts(normalize=True)` (proportions), not raw counts.
  - Confirmed correlation values are plausible (max |r| = 0.146 for numerical features, consistent with known low linear correlation in this dataset).

---

## Entry 3
- **Date/Time**: 2026-03-17T00:00:00Z
- **Goal**: Train and evaluate a Logistic Regression classifier on the UCI Adult Income dataset
- **User Request**: Write `t3_model.py` that loads/preprocesses data, performs a stratified 80/20 split (random_state=42), trains LogisticRegression(max_iter=1000), and prints accuracy, F1, precision, recall, and confusion matrix. Must be fully reproducible.
- **Files Changed**:
  - `workspace_claude/t3_model.py` (created)
- **Summary of Edits**:
  - **Missing value handling**: Stripped whitespace from all string columns, then dropped 3,620 rows containing `'?'` sentinel → 45,222 clean rows (consistent with T1/T2).
  - **Target encoding**: `'>50K'` → 1, `'<=50K'` → 0 (binary int).
  - **Feature split**: 6 numerical features (`age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`) and 8 categorical features (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`). `education` retained alongside `education-num` for OHE completeness; model selects relevant signal automatically.
  - **Stratified split**: `train_test_split(test_size=0.20, stratify=y, random_state=42)` → 36,177 train / 9,045 test. Test class balance: 75.2% ≤50K / 24.8% >50K — matches full dataset (preserves class proportion as required).
  - **Preprocessing pipeline**: `ColumnTransformer` with `StandardScaler` on numericals and `OneHotEncoder(handle_unknown='ignore')` on categoricals, wrapped in a `sklearn.pipeline.Pipeline`.
  - **Model**: `LogisticRegression(max_iter=1000, random_state=42)` — converged within iteration budget.
  - **Evaluation printed**: Accuracy 0.8450, Precision 0.7344, Recall 0.5870, F1 0.6525, plus confusion matrix and full `classification_report`.
- **Verification Performed**:
  - Ran `python3 t3_model.py` from `workspace_claude/`; zero errors, zero warnings.
  - **Reproducibility**: Ran the script three times and confirmed identical output via `md5` hash (`a2b8b4ea3b3fa63948a67988c5886f58`) on all runs.
  - **Stratification check**: Test set class balance (75.2% / 24.8%) matches full-dataset balance confirmed in T2, confirming stratify worked correctly.
  - **Statistical reasoning**: Accuracy (0.845) is a reasonable ceiling for logistic regression on this dataset (linear decision boundary); F1 for the minority class (>50K) is 0.65, reflecting the 3:1 class imbalance — recall (0.587) is lower than precision (0.734), meaning the model is conservative about predicting >50K. This is expected for an untuned linear model on an imbalanced target.
  - Confirmed clean row count (45,222) is consistent with T1 and T2.

---

## Entry 4
- **Date/Time**: 2026-03-17T00:00:00Z
- **Goal**: Identify silent bugs in broken_pipeline.py, produce a corrected script, and quantify the accuracy difference
- **User Request**: Identify all bugs in broken_pipeline.py (explaining what is wrong and the statistical consequence), produce a corrected t4_fixed_pipeline.py with inline correction comments, run both scripts, and report the accuracy difference.
- **Files Changed**:
  - `workspace_claude/t4_fixed_pipeline.py` (created)
  - `workspace_claude/AGENT_LOG.md` (updated)
- **Summary of Edits**:

  **Bug 1 — Arbitrary ordinal encoding of 'workclass' (lines 25–35)**
  - *What is wrong*: A hand-crafted integer map (Private→3, Federal-gov→0, Never-worked→7, etc.) was applied to `workclass`. `workclass` is a *nominal* categorical variable with no natural numeric ordering.
  - *Statistical consequence*: Logistic regression receives a single numeric column and fits one slope coefficient to it, treating the integers as an ordered continuous scale. The model then assumes, for example, that `Never-worked (7)` is 7× greater than `Federal-gov (0)` — a relationship that has no statistical meaning. This biases the learned coefficient for workclass and distorts its contribution to the log-odds.
  - *Fix*: Replaced the ordinal map with `pd.get_dummies(df, columns=['workclass'], drop_first=True)`. One-hot encoding creates a binary indicator per category, letting the model estimate an independent coefficient for each level — the correct treatment for a nominal variable in a linear model.

  **Bug 2 — LabelEncoder applied to remaining nominal categoricals (lines 42–44)**
  - *What is wrong*: `LabelEncoder` was used inside a loop to encode `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, and `native-country`. LabelEncoder assigns arbitrary, alphabetically-sorted integer codes to categories.
  - *Statistical consequence*: Identical to Bug 1. Logistic regression treats each encoded column as a single continuous ordinal variable and fits one slope. The arbitrary code differences (e.g., `Masters - Bachelors = 4`) are treated as meaningful numeric distances, distorting all seven categorical coefficients and misrepresenting the true relationship between each categorical feature and the target.
  - *Fix*: Replaced `LabelEncoder` loop with `pd.get_dummies(df, columns=cat_cols, drop_first=True)` for all seven columns, applying the same one-hot encoding rationale as Bug 1.

  **Bug 3 — Data leakage: StandardScaler fitted on entire dataset before train/test split (lines 56–62)**
  - *What is wrong*: `scaler.fit_transform(X[numeric_cols])` was called on all 45,222 rows *before* the `train_test_split` call. The scaler computes column means and standard deviations from the full dataset — test-set rows included.
  - *Statistical consequence*: Test-set statistics contaminate the normalisation of every training sample. This is a form of preprocessing leakage: the model training pipeline has indirectly "seen" the test set, violating the independence of the holdout evaluation and producing optimistically biased accuracy/F1 estimates.
  - *Fix*: Moved `train_test_split` to before the scaler. `scaler.fit_transform()` is then called only on `X_train`; `scaler.transform()` (no fit) is called on `X_test`. The test set is normalised exclusively using training-set statistics, preserving full holdout independence.

- **Verification Performed**:
  - Ran `python3 broken_pipeline.py` → zero exceptions; output captured.
  - Ran `python3 t4_fixed_pipeline.py` → zero exceptions; output captured.
  - **Statistical reasoning for observed improvement**: Fixing the encoding bugs converts 7 columns (plus workclass) from single ordinal features to proper OHE feature sets. This gives logistic regression the correct feature space to learn linear decision boundaries for nominal categoricals, substantially improving recall for the minority class (>50K) from 0.4465 → 0.5879, reflecting that the model can now correctly separate categories rather than fitting a meaningless numeric slope. The leakage fix has a smaller effect on accuracy but ensures metrics are honest estimates of generalisation performance.

  | Metric    | broken_pipeline.py | t4_fixed_pipeline.py | Δ        |
  |-----------|-------------------|---------------------|----------|
  | Accuracy  | 0.8189            | 0.8450              | +0.0261  |
  | F1-Score  | 0.5500            | 0.6528              | +0.1028  |
  | Precision | 0.7160            | 0.7339              | +0.0179  |
  | Recall    | 0.4465            | 0.5879              | +0.1414  |

  The fixed pipeline improves accuracy by **+2.61 percentage points** and F1 for the minority class by **+10.28 percentage points**. The large recall gain (+14.14 pp) confirms that the arbitrary ordinal encoding was the dominant bug: it caused the model to be severely under-sensitive to the `>50K` class by forcing logistic regression to fit a single slope to meaningless integer codes.

---

