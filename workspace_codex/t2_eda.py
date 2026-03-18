from __future__ import annotations

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = Path("adult_income.csv")
OUTPUT_DIR = Path("t2_eda_outputs")
TARGET_COLUMN = "income"
MISSING_TOKEN = "?"


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def load_and_clean_data() -> tuple[pd.DataFrame, int, int]:
    df = pd.read_csv(DATA_PATH, dtype="string", keep_default_na=False)
    initial_rows = len(df)
    encoded_missing_mask = df.eq(MISSING_TOKEN).any(axis=1)
    rows_removed = int(encoded_missing_mask.sum())
    cleaned = df.mask(df.eq(MISSING_TOKEN), pd.NA).dropna().copy()

    for column in cleaned.columns:
        numeric_series = pd.to_numeric(cleaned[column], errors="coerce")
        if numeric_series.notna().all():
            cleaned[column] = numeric_series

    return cleaned, rows_removed, initial_rows


def split_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = [
        column for column in df.columns if column != TARGET_COLUMN and pd.api.types.is_numeric_dtype(df[column])
    ]
    categorical_features = [
        column for column in df.columns if column != TARGET_COLUMN and not pd.api.types.is_numeric_dtype(df[column])
    ]
    return numeric_features, categorical_features


def save_continuous_distribution_plots(df: pd.DataFrame, numeric_features: list[str]) -> list[Path]:
    output_paths: list[Path] = []
    income_order = ["<=50K", ">50K"]
    colors = {"<=50K": "#4C78A8", ">50K": "#F58518"}

    for feature in numeric_features:
        fig, ax = plt.subplots(figsize=(10, 6))
        feature_data = df[feature]
        bin_count = 30 if feature_data.nunique() > 30 else max(10, int(feature_data.nunique()))

        for income_value in income_order:
            subset = df.loc[df[TARGET_COLUMN] == income_value, feature]
            ax.hist(
                subset,
                bins=bin_count,
                density=True,
                alpha=0.5,
                label=f"{income_value} density",
                color=colors[income_value],
            )

        ax.axvline(feature_data.mean(), color="black", linestyle="--", linewidth=1.5, label="Overall mean")
        ax.set_title(f"Distribution of {feature} by Income Group")
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()

        output_path = OUTPUT_DIR / f"distribution_{slugify(feature)}.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def save_categorical_frequency_plots(df: pd.DataFrame, categorical_features: list[str]) -> list[Path]:
    output_paths: list[Path] = []

    for feature in categorical_features + [TARGET_COLUMN]:
        distribution = df[feature].value_counts(normalize=True).sort_values(ascending=True)

        fig_height = max(4, 0.35 * len(distribution) + 2)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        ax.barh(distribution.index.astype(str), distribution.values, color="#54A24B", label="Sample proportion")
        ax.set_title(f"Frequency Distribution of {feature}")
        ax.set_xlabel("Proportion of Cleaned Sample")
        ax.set_ylabel(feature)
        ax.set_xlim(0, max(0.05, distribution.max() * 1.1))
        ax.legend()
        fig.tight_layout()

        output_path = OUTPUT_DIR / f"frequency_{slugify(feature)}.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def save_grouped_categorical_proportion_plots(df: pd.DataFrame, categorical_features: list[str]) -> list[Path]:
    output_paths: list[Path] = []
    income_order = ["<=50K", ">50K"]
    colors = ["#4C78A8", "#F58518"]

    for feature in categorical_features:
        proportions = pd.crosstab(df[feature], df[TARGET_COLUMN], normalize="index")
        proportions = proportions.reindex(columns=income_order, fill_value=0)
        positive_rate = proportions[">50K"] if ">50K" in proportions.columns else pd.Series(0, index=proportions.index)
        proportions = proportions.loc[positive_rate.sort_values().index]

        fig_height = max(4, 0.35 * len(proportions) + 2)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        left = np.zeros(len(proportions))

        for income_value, color in zip(income_order, colors):
            values = proportions[income_value].to_numpy()
            ax.barh(
                proportions.index.astype(str),
                values,
                left=left,
                color=color,
                label=f"{income_value} proportion",
            )
            left += values

        ax.set_title(f"Income Proportions within {feature} Categories")
        ax.set_xlabel("Proportion within Category")
        ax.set_ylabel(feature)
        ax.set_xlim(0, 1)
        ax.legend()
        fig.tight_layout()

        output_path = OUTPUT_DIR / f"grouped_proportion_{slugify(feature)}.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def save_correlation_heatmap(df: pd.DataFrame, numeric_features: list[str]) -> Path:
    correlation = df[numeric_features].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_features)))
    ax.set_yticks(range(len(numeric_features)))
    ax.set_xticklabels(numeric_features, rotation=45, ha="right")
    ax.set_yticklabels(numeric_features)
    ax.set_title("Correlation Heatmap for Numerical Features")

    for row in range(len(numeric_features)):
        for col in range(len(numeric_features)):
            ax.text(col, row, f"{correlation.iloc[row, col]:.2f}", ha="center", va="center", color="black")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Pearson Correlation")
    fig.tight_layout()

    output_path = OUTPUT_DIR / "correlation_heatmap.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_narrative_summary(df: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]) -> str:
    class_balance = df[TARGET_COLUMN].value_counts(normalize=True).mul(100)

    numeric_diffs: list[tuple[str, float]] = []
    grouped_numeric_means = {}
    for feature in numeric_features:
        means = df.groupby(TARGET_COLUMN)[feature].mean()
        grouped_numeric_means[feature] = means
        std = df[feature].std(ddof=0)
        standardized_diff = float((means[">50K"] - means["<=50K"]) / std) if std else 0.0
        numeric_diffs.append((feature, standardized_diff))

    numeric_diffs.sort(key=lambda item: abs(item[1]), reverse=True)

    education_rates = pd.crosstab(df["education"], df[TARGET_COLUMN], normalize="index")[">50K"].sort_values()
    relationship_rates = pd.crosstab(df["relationship"], df[TARGET_COLUMN], normalize="index")[">50K"].sort_values()
    capital_gain_nonzero = (
        df.assign(nonzero_capital_gain=(df["capital-gain"] > 0))
        .groupby(TARGET_COLUMN)["nonzero_capital_gain"]
        .mean()
        .mul(100)
    )

    summary_lines = [
        "EDA Narrative Summary",
        "-" * 80,
        (
            f"Rows after dropping non-standard missing values ('?'): {len(df):,}. "
            f"Class balance remains skewed: {class_balance['<=50K']:.2f}% of records are '<=50K' "
            f"and {class_balance['>50K']:.2f}% are '>50K'."
        ),
        (
            "Pattern 1: Education is the strongest broad predictor in the cleaned sample. "
            f"'education-num' shows the largest standardized mean gap between income classes "
            f"({numeric_diffs[0][1]:.2f} standard deviations), and the >50K rate ranges from "
            f"{education_rates.iloc[0] * 100:.1f}% for '{education_rates.index[0]}' to "
            f"{education_rates.iloc[-1] * 100:.1f}% for '{education_rates.index[-1]}'."
        ),
        (
            "Pattern 2: Work intensity and capital-related variables should matter materially for prediction. "
            f"Adults earning >50K work {grouped_numeric_means['hours-per-week']['>50K']:.2f} hours per week on average "
            f"versus {grouped_numeric_means['hours-per-week']['<=50K']:.2f} for the <=50K group, and "
            f"{capital_gain_nonzero['>50K']:.2f}% of >50K records have non-zero capital gain compared with "
            f"{capital_gain_nonzero['<=50K']:.2f}% in the <=50K group."
        ),
        (
            "Pattern 3: Household and social-position features create strong categorical separation. "
            f"The >50K rate is {relationship_rates.iloc[-1] * 100:.1f}% for '{relationship_rates.index[-1]}' "
            f"but only {relationship_rates.iloc[0] * 100:.1f}% for '{relationship_rates.index[0]}', "
            "which suggests relationship and marital-status variables can add substantial signal."
        ),
        (
            "Model training impact: the target is materially imbalanced, so evaluation should not rely on accuracy alone. "
            "Stratified splits and metrics such as PR-AUC, ROC-AUC, balanced accuracy, or class-specific recall will give a more faithful view of performance on the >50K class."
        ),
    ]
    return "\n".join(summary_lines)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df, rows_removed, initial_rows = load_and_clean_data()
    numeric_features, categorical_features = split_feature_types(df)

    continuous_plots = save_continuous_distribution_plots(df, numeric_features)
    categorical_plots = save_categorical_frequency_plots(df, categorical_features)
    grouped_plots = save_grouped_categorical_proportion_plots(df, categorical_features)
    heatmap_path = save_correlation_heatmap(df, numeric_features)
    narrative_summary = build_narrative_summary(df, numeric_features, categorical_features)

    summary_path = OUTPUT_DIR / "narrative_summary.txt"
    summary_path.write_text(narrative_summary + "\n", encoding="utf-8")

    print(f"Loaded dataset: {DATA_PATH}")
    print(f"Initial rows: {initial_rows:,}")
    print(f"Rows removed due to '{MISSING_TOKEN}' placeholders: {rows_removed}")
    print(f"Rows used for EDA: {len(df):,}")
    print(f"Continuous features plotted: {', '.join(numeric_features)}")
    print(f"Categorical features plotted: {', '.join(categorical_features + [TARGET_COLUMN])}")
    print(f"Grouped categorical proportion plots created for: {', '.join(categorical_features)}")
    print(f"Correlation heatmap saved to: {heatmap_path}")
    print(f"Narrative summary saved to: {summary_path}")
    print(f"Plot output directory: {OUTPUT_DIR}")
    print(f"Number of plot files created: {len(continuous_plots) + len(categorical_plots) + len(grouped_plots) + 1}")
    print("\nSaved plot files:")
    for path in continuous_plots + categorical_plots + grouped_plots + [heatmap_path]:
        print(f" - {path}")
    print("\n" + narrative_summary)


if __name__ == "__main__":
    main()
