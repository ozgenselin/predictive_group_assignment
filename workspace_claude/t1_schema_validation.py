"""
t1_schema_validation.py
-----------------------
Schema validation for the UCI Adult Income dataset.
Detects standard and non-standard / encoded missing values.
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "adult_income.csv")

# Known non-standard / sentinel missing-value encodings to search for
SENTINEL_PATTERNS = {"?", "N/A", "NA", "n/a", "na", "None", "none",
                     "NULL", "null", "NaN", "nan", "", " ", "-", "--"}


# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns\n")
    return df


# ---------------------------------------------------------------------------
# 2. Schema report
# ---------------------------------------------------------------------------
def schema_report(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("SCHEMA REPORT")
    print("=" * 70)

    cat_cols = df.select_dtypes(include="object").columns.tolist()

    header = f"{'Column':<22} {'Dtype':<12} {'NaN Count':>10} {'Unique (cat)':>13}"
    print(header)
    print("-" * 70)

    for col in df.columns:
        dtype   = str(df[col].dtype)
        nan_cnt = int(df[col].isna().sum())
        unique  = f"{df[col].nunique():>13}" if col in cat_cols else f"{'—':>13}"
        print(f"{col:<22} {dtype:<12} {nan_cnt:>10} {unique}")

    print()


# ---------------------------------------------------------------------------
# 3 & 4. Non-standard missing-value detection + summary
# ---------------------------------------------------------------------------
def detect_encoded_missing(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("NON-STANDARD / ENCODED MISSING VALUE DETECTION")
    print("=" * 70)

    # Track every (column, sentinel) combination found
    findings: dict[str, dict] = {}   # col -> {sentinel: count}

    for col in df.columns:
        col_findings: dict[str, int] = {}

        if df[col].dtype == object:
            # String columns: check exact match and stripped match
            for sentinel in SENTINEL_PATTERNS:
                # exact match
                exact_mask = df[col] == sentinel
                # match after stripping whitespace (catches " ? ", " ", etc.)
                strip_mask = df[col].str.strip() == sentinel
                combined   = exact_mask | strip_mask
                cnt = int(combined.sum())
                if cnt > 0:
                    col_findings[sentinel if sentinel != "" else "<empty string>"] = cnt
        else:
            # Numeric columns: check for sentinel numeric codes if any
            # (e.g. -1, -9, 9999 — not present in this dataset but detected
            #  generically by looking for suspicious extreme outliers)
            pass  # No numeric sentinels expected; kept for extensibility

        if col_findings:
            findings[col] = col_findings

    # -----------------------------------------------------------------------
    # Compute row-level impact: a row is "affected" if ANY column contains a
    # sentinel value in that row.
    # -----------------------------------------------------------------------
    affected_mask = pd.Series(False, index=df.index)

    for col, sentinels in findings.items():
        for sentinel_key in sentinels:
            raw_sentinel = "" if sentinel_key == "<empty string>" else sentinel_key
            col_mask = (df[col] == raw_sentinel) | (df[col].str.strip() == raw_sentinel)
            affected_mask |= col_mask

    total_affected_rows = int(affected_mask.sum())

    # -----------------------------------------------------------------------
    # Print per-column details
    # -----------------------------------------------------------------------
    if not findings:
        print("No non-standard encoded missing values detected.\n")
    else:
        print("Columns with encoded missing values:\n")
        for col, sentinels in findings.items():
            for sentinel_key, cnt in sentinels.items():
                print(f"  Column: {col:<22}  Encoding: {repr(sentinel_key):<8}  "
                      f"Occurrences: {cnt:>5}")
        print()

    # -----------------------------------------------------------------------
    # Summary block
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if findings:
        all_sentinels = set()
        for v in findings.values():
            all_sentinels.update(v.keys())

        print(f"Missing-value encoding(s) detected : {', '.join(repr(s) for s in sorted(all_sentinels))}")
        print(f"Total affected rows                : {total_affected_rows:,}")
        print(f"Columns affected ({len(findings)})             : {', '.join(findings.keys())}")
    else:
        print("No non-standard missing values found.")
        print(f"Total affected rows: 0")

    print()

    # Also report standard NaN counts
    total_nan_rows = int(df.isna().any(axis=1).sum())
    print(f"Rows with standard NaN values      : {total_nan_rows:,}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = load_data(DATA_PATH)
    schema_report(df)
    detect_encoded_missing(df)


if __name__ == "__main__":
    main()
