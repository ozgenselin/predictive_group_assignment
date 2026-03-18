"""
t2_eda.py — Exploratory Data Analysis: UCI Adult Income Dataset
================================================================
Workflow:
  1. Load CSV and drop rows with '?' sentinel values (identified in Task 1).
  2. Distribution plots for all continuous features.
  3. Frequency / proportion plots for all categorical features.
  4. Correlation heatmap for numerical features.
  5. Grouped categorical comparisons (proportions, not raw counts).
  6. Print a written narrative summary.

All figures are saved to workspace_claude/eda_plots/.
"""

import os
import pathlib
import textwrap

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────────────────
HERE      = pathlib.Path(__file__).parent.resolve()
CSV_PATH  = HERE / "adult_income.csv"
PLOT_DIR  = HERE / "eda_plots"
PLOT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE   = {"<=50K": "#4C72B0", ">50K": "#DD8452"}

# ── 1. Load & clean ──────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1 — Loading data and handling encoded missing values ('?')")
print("=" * 70)

df_raw = pd.read_csv(CSV_PATH)
print(f"  Raw shape : {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

# Strip leading/trailing whitespace from all string columns
for col in df_raw.select_dtypes(include="object").columns:
    df_raw[col] = df_raw[col].str.strip()

# Drop rows containing '?' in ANY column (Task 1 identified workclass,
# occupation, native-country as affected columns)
sentinel_cols = ["workclass", "occupation", "native-country"]
mask_bad = (df_raw[sentinel_cols] == "?").any(axis=1)
df = df_raw[~mask_bad].copy().reset_index(drop=True)

print(f"  Rows with '?' sentinel : {mask_bad.sum():,}")
print(f"  Clean shape            : {df.shape[0]:,} rows × {df.shape[1]} columns\n")

# ── Column taxonomy ──────────────────────────────────────────────────────────
CONTINUOUS   = ["age", "fnlwgt", "education-num",
                "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL  = ["workclass", "education", "marital-status", "occupation",
                "relationship", "race", "sex", "native-country"]
TARGET       = "income"

# ── 2. Distribution plots — continuous features ──────────────────────────────
print("=" * 70)
print("STEP 2 — Distribution plots for continuous features")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(CONTINUOUS):
    ax = axes[i]
    for label, grp in df.groupby(TARGET)[col]:
        ax.hist(grp, bins=40, alpha=0.65, label=label,
                color=PALETTE[label], edgecolor="white", linewidth=0.4)
    ax.set_title(f"Distribution of {col}", fontsize=13, fontweight="bold")
    ax.set_xlabel(col, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(title="Income", fontsize=9)

fig.suptitle("Continuous Feature Distributions by Income Group",
             fontsize=15, fontweight="bold", y=1.01)
fig.tight_layout()
out = PLOT_DIR / "01_continuous_distributions.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {out.name}")

# ── 3. Frequency / proportion plots — categorical features ───────────────────
print("\n" + "=" * 70)
print("STEP 3 — Frequency plots for categorical features")
print("=" * 70)

# 3a  Raw frequency bar charts (one per categorical column)
for col in CATEGORICAL:
    order = df[col].value_counts().index.tolist()
    fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.9), 5))
    counts = df[col].value_counts().reindex(order)
    ax.bar(range(len(order)), counts.values,
           color=sns.color_palette("muted", len(order)), edgecolor="white")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=9)
    ax.set_title(f"Frequency — {col}", fontsize=13, fontweight="bold")
    ax.set_xlabel(col, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    safe = col.replace("-", "_")
    out  = PLOT_DIR / f"02_freq_{safe}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}")

# 3b  Target variable (income) frequency
fig, ax = plt.subplots(figsize=(6, 4))
counts  = df[TARGET].value_counts()
bars    = ax.bar(counts.index, counts.values,
                 color=[PALETTE[k] for k in counts.index], edgecolor="white", width=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
            f"{val:,}\n({val/len(df):.1%})", ha="center", va="bottom", fontsize=10)
ax.set_title("Target Variable Distribution — income", fontsize=13, fontweight="bold")
ax.set_xlabel("Income Class", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.tight_layout()
out = PLOT_DIR / "03_target_distribution.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {out.name}")

# ── 4. Correlation heatmap — numerical features ──────────────────────────────
print("\n" + "=" * 70)
print("STEP 4 — Correlation heatmap (numerical features)")
print("=" * 70)

num_df  = df[CONTINUOUS].copy()
corr    = num_df.corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask    = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, linewidths=0.5, ax=ax,
            annot_kws={"size": 10})
ax.set_title("Pearson Correlation — Numerical Features",
             fontsize=13, fontweight="bold")
fig.tight_layout()
out = PLOT_DIR / "04_correlation_heatmap.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {out.name}")
print(f"\n  Top pairwise correlations (|r| > 0.10):")
corr_pairs = (corr.where(~mask).stack()
              .rename("r").reset_index()
              .rename(columns={"level_0": "feat_a", "level_1": "feat_b"})
              .assign(abs_r=lambda x: x["r"].abs())
              .sort_values("abs_r", ascending=False))
for _, row in corr_pairs[corr_pairs["abs_r"] > 0.10].iterrows():
    print(f"    {row.feat_a:18s} ↔ {row.feat_b:18s}  r = {row.r:+.3f}")

# ── 5. Grouped categorical comparisons — proportions ────────────────────────
print("\n" + "=" * 70)
print("STEP 5 — Grouped categorical comparisons (proportions)")
print("=" * 70)

# Focus on the most informative categoricals for prediction
KEY_CATS = ["sex", "marital-status", "education", "workclass",
            "occupation", "relationship"]

for col in KEY_CATS:
    # Proportion of >50K within each category level
    prop = (df.groupby(col)[TARGET]
              .value_counts(normalize=True)
              .unstack(fill_value=0)
              .reindex(columns=["<=50K", ">50K"])
              .sort_values(">50K", ascending=False))

    order  = prop.index.tolist()
    x      = np.arange(len(order))
    width  = 0.55

    fig, ax = plt.subplots(figsize=(max(8, len(order) * 1.1), 5))
    ax.bar(x, prop["<=50K"].values * 100, width,
           label="<=50K", color=PALETTE["<=50K"], edgecolor="white")
    ax.bar(x, prop[">50K"].values  * 100, width,
           bottom=prop["<=50K"].values * 100,
           label=">50K",  color=PALETTE[">50K"],  edgecolor="white")

    # annotate >50K proportion on each bar
    for xi, (_, row) in zip(x, prop.iterrows()):
        pct = row[">50K"] * 100
        if pct > 3:
            ax.text(xi, 100 - pct / 2, f"{pct:.0f}%",
                    ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title(f"Income Proportion by {col}  (sorted by >50K rate)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel(col, fontsize=11)
    ax.set_ylabel("Proportion (%)", fontsize=11)
    ax.legend(title="Income", fontsize=9)
    fig.tight_layout()
    safe = col.replace("-", "_")
    out  = PLOT_DIR / f"05_grouped_{safe}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}")

# ── 6. Narrative summary ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6 — Narrative Summary")
print("=" * 70)

n_total   = len(df)
vc        = df[TARGET].value_counts()
n_neg     = vc["<=50K"]
n_pos     = vc[">50K"]
ratio     = n_neg / n_pos

# ── Class balance stats
print()
print(textwrap.dedent(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                  EDA NARRATIVE SUMMARY                              │
├─────────────────────────────────────────────────────────────────────┤
│  Dataset (after dropping '?' rows): {n_total:,} observations             │
└─────────────────────────────────────────────────────────────────────┘

── CLASS BALANCE ──────────────────────────────────────────────────────
  <=50K  : {n_neg:,}  ({n_neg/n_total:.1%})
  >50K   : {n_pos:,}  ({n_pos/n_total:.1%})
  Imbalance ratio (negative : positive) ≈ {ratio:.1f} : 1

  The target is moderately imbalanced — roughly 3 out of 4 individuals
  earn <=50K. This imbalance will inflate raw accuracy scores and bias
  models toward the majority class. Evaluation should rely on metrics
  that are robust to imbalance: F1-score, ROC-AUC, or precision-recall
  AUC rather than accuracy alone. Techniques such as stratified
  cross-validation, class weighting, or resampling (SMOTE / under-
  sampling) should be considered during model training.
"""))

# ── Compute key statistics to inform patterns
# Pattern 1: education-num vs income
edu_hi = df[df["education-num"] >= 13][TARGET].value_counts(normalize=True)[">50K"]
edu_lo = df[df["education-num"] <  10][TARGET].value_counts(normalize=True)[">50K"]

# Pattern 2: marital-status
ms_rates = (df.groupby("marital-status")[TARGET]
              .value_counts(normalize=True).unstack()[">50K"]
              .sort_values(ascending=False))
top_ms   = ms_rates.index[0]
top_ms_r = ms_rates.iloc[0]

# Pattern 3: capital-gain
cg_nonzero = df[df["capital-gain"] > 0]
cg_rate    = (cg_nonzero[TARGET] == ">50K").mean()
cg_pct     = (df["capital-gain"] > 0).mean()

# Pattern 4: age
age_hi = df[df["age"] >= 40][TARGET].value_counts(normalize=True)[">50K"]
age_lo = df[df["age"] <  30][TARGET].value_counts(normalize=True)[">50K"]

print(textwrap.dedent(f"""
── THREE MOST IMPORTANT PATTERNS FOR PREDICTIVE MODELLING ────────────

  PATTERN 1 — Education level (education-num) is a strong predictor
  ──────────────────────────────────────────────────────────────────
  Individuals with education-num ≥ 13 (Bachelor's degree or higher)
  earn >50K at a rate of {edu_hi:.1%}, compared with only {edu_lo:.1%} for
  those with education-num < 10 (below some-college level). The
  continuous feature 'education-num' and the correlated categorical
  'education' together offer one of the cleanest decision boundaries
  in the dataset.  Model impact: tree-based models are likely to
  place early splits on this feature; linear models should benefit
  from polynomial or threshold encoding of education-num.

  PATTERN 2 — Marital status captures economic household structure
  ──────────────────────────────────────────────────────────────────
  '{top_ms}' has the highest >50K rate at {top_ms_r:.1%}. Marital
  status acts as a proxy for dual-income households and career stage.
  The 'relationship' feature (Husband, Wife, Own-child …) encodes
  overlapping information, creating strong multicollinearity between
  these two categorical features. Care should be taken to avoid
  including both in linear models without regularisation, though
  tree-based models will handle the redundancy naturally.

  PATTERN 3 — Capital gains/losses are highly discriminative but sparse
  ──────────────────────────────────────────────────────────────────────
  Only {cg_pct:.1%} of individuals report non-zero capital-gain, yet
  among those who do, {cg_rate:.1%} earn >50K. This extreme sparsity
  combined with high signal means that a binary 'has_capital_activity'
  indicator may outperform the raw continuous value for many models.
  The right-skewed distribution also warrants a log1p transform to
  prevent gradient-based learners from over-weighting large outliers.

── ADDITIONAL OBSERVATION ────────────────────────────────────────────
  Age shows a monotonic positive trend with income: the >50K rate
  rises from {age_lo:.1%} for under-30s to {age_hi:.1%} for 40+, reflecting
  career progression. Combined with education-num and marital-status,
  these three features alone provide substantial predictive signal and
  should be prioritised in feature importance analyses.

  Sex-based disparity is also visible: the grouped proportion plot
  shows males earning >50K at a meaningfully higher rate than females.
  This demographic imbalance is present in the raw data and models
  trained on it may propagate this bias — fairness-aware evaluation
  metrics are recommended for any deployed application.
"""))

print("=" * 70)
print(f"All plots saved to: {PLOT_DIR}")
print("EDA complete.")
print("=" * 70)
