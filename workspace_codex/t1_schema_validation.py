from __future__ import annotations

import pandas as pd
from pandas.api.types import is_string_dtype


SUSPICIOUS_MISSING_TOKENS = {
    "",
    "?",
    "NA",
    "N/A",
    "NaN",
    "nan",
    "NULL",
    "Null",
    "null",
    "None",
    "none",
    "missing",
    "Missing",
}


def detect_encoded_missing_values(df: pd.DataFrame) -> tuple[dict[str, dict[str, int]], set[str]]:
    encoded_missing_by_column: dict[str, dict[str, int]] = {}
    affected_rows: set[str] = set()

    for column in df.columns:
        series = df[column].astype("string")
        normalized = series.fillna("").str.strip()
        counts = normalized.value_counts(dropna=False)

        found_tokens = {
            token: int(counts.get(token, 0))
            for token in SUSPICIOUS_MISSING_TOKENS
            if int(counts.get(token, 0)) > 0
        }

        if found_tokens:
            encoded_missing_by_column[column] = dict(sorted(found_tokens.items()))
            affected_rows.update(df.index[normalized.isin(found_tokens.keys())].astype(str).tolist())

    return encoded_missing_by_column, affected_rows


def infer_clean_schema(df: pd.DataFrame, encoded_missing_by_column: dict[str, dict[str, int]]) -> pd.DataFrame:
    cleaned_df = df.copy()

    for column, token_counts in encoded_missing_by_column.items():
        tokens = list(token_counts.keys())
        cleaned_df[column] = cleaned_df[column].astype("string").str.strip().replace(tokens, pd.NA)

    for column in cleaned_df.columns:
        cleaned_df[column] = cleaned_df[column].convert_dtypes()
        if is_string_dtype(cleaned_df[column]):
            numeric = pd.to_numeric(cleaned_df[column], errors="coerce")
            non_missing = cleaned_df[column].notna()
            if non_missing.any() and numeric[non_missing].notna().all():
                cleaned_df[column] = numeric.astype("Int64")

    return cleaned_df


def build_schema_report(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for column in cleaned_df.columns:
        series = cleaned_df[column]
        is_categorical = is_string_dtype(series)
        records.append(
            {
                "column": column,
                "inferred_dtype": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "unique_values": int(series.nunique(dropna=True)) if is_categorical else "N/A",
            }
        )

    return pd.DataFrame(records)


def print_encoded_missing_report(encoded_missing_by_column: dict[str, dict[str, int]], affected_row_count: int) -> None:
    print("\nEncoded Missing Value Detection")
    print("-" * 80)

    if not encoded_missing_by_column:
        print("No encoded missing values were detected beyond pandas-native nulls or blank strings.")
        return

    for column, token_counts in encoded_missing_by_column.items():
        token_summary = ", ".join(f"{repr(token)} ({count})" for token, count in token_counts.items())
        print(f"{column}: {token_summary}")

    all_tokens = sorted({token for token_counts in encoded_missing_by_column.values() for token in token_counts})
    print("\nSummary")
    print("-" * 80)
    print(f"Missing value encoding used: {', '.join(repr(token) for token in all_tokens)}")
    print(f"Total rows affected: {affected_row_count}")
    print(f"Columns containing encoded missing values: {', '.join(encoded_missing_by_column.keys())}")


def main() -> None:
    dataset_path = "adult_income.csv"
    raw_df = pd.read_csv(dataset_path, dtype="string", keep_default_na=False)

    encoded_missing_by_column, affected_rows = detect_encoded_missing_values(raw_df)
    cleaned_df = infer_clean_schema(raw_df, encoded_missing_by_column)
    schema_report = build_schema_report(cleaned_df)

    print(f"Loaded dataset: {dataset_path}")
    print(f"Row count: {len(cleaned_df):,}")
    print(f"Column count: {len(cleaned_df.columns)}")
    print("\nSchema Report")
    print("-" * 80)
    print(schema_report.to_string(index=False))

    print_encoded_missing_report(encoded_missing_by_column, len(affected_rows))


if __name__ == "__main__":
    main()
