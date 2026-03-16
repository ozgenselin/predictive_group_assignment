import pandas as pd

def validate_schema(file_path):
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)

    print("\n" + "="*80)
    print(" SCHEMA REPORT")
    print("="*80)
    
    # Standard missing values
    standard_missing = df.isnull().sum()
    
    print(f"{'Column Name':<20} | {'Data Type':<10} | {'Standard Missing':<18} | {'Unique Values (Cat)'}")
    print("-" * 80)
    for col in df.columns:
        # FIX ISSUE 1: Extract real pandas dtype name (e.g., 'object', 'int64', 'category')
        dtype_name = getattr(df[col].dtype, 'name', str(df[col].dtype))
        if dtype_name == 'str':
             dtype_name = 'object' # fallback to pandas terminology for raw python strings

        missing_count = standard_missing[col]
        
        # We only count unique items for non-numeric columns like object, string, and category
        is_numeric = pd.api.types.is_numeric_dtype(df[col].dtype)
        if not is_numeric:
            unique_count = str(df[col].nunique(dropna=True))
        else:
            unique_count = "N/A"
            
        print(f"{col:<20} | {dtype_name:<10} | {missing_count:<18} | {unique_count}")

    print("\n" + "="*80)
    print(" NON-STANDARD MISSING VALUES DETECTION")
    print("="*80)
    
    non_standard_missing = {}
    affected_rows_idx = set()
    
    # FIX ISSUE 2: Do not assume `dtype == 'object'`.
    # Many string columns might show up as 'string', 'category', or have other types.
    # To be safe, scan all non-numeric columns.
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col].dtype):
            # get value frequencies
            val_counts = df[col].value_counts(dropna=False)
            for val, count in val_counts.items():
                if isinstance(val, str) or pd.api.types.is_string_dtype(type(val)):
                    val_str = str(val)
                    stripped_val = val_str.strip()
                    # Commonly seen missing placeholders:
                    if stripped_val in ['?', 'Unknown', 'unknown', 'N/A', 'n/a', 'NA', 'na', '', '-']:
                        if col not in non_standard_missing:
                            non_standard_missing[col] = {}
                        non_standard_missing[col][val_str] = count
                        
                        # Find indices exactly where it matches (after strip if strip was applied, to allow ' ?' -> '?')
                        # For performance & robustness, string map back:
                        # (Doing a boolean mask)
                        mask = df[col].astype(str).str.strip() == stripped_val
                        idx = df[mask].index
                        affected_rows_idx.update(idx)

    if not non_standard_missing:
        print("No non-standard missing values detected in the dataset.\n")
    else:
        for col, missing_dict in non_standard_missing.items():
            for val, count in missing_dict.items():
                print(f"Column '{col}' contains {count} occurrences of {repr(val)}")
        
    print("\n" + "="*80)
    print(" SUMMARY OF MISSING VALUES")
    print("="*80)
    if non_standard_missing:
        all_encodings = set()
        for col_dict in non_standard_missing.values():
            for encoding in col_dict.keys():
                all_encodings.add(repr(encoding))
        
        encodings_str = ", ".join(all_encodings)
        affected_cols = list(non_standard_missing.keys())
        affected_cols_str = ", ".join(affected_cols)
        total_affected_rows = len(affected_rows_idx)
        
        print(f"Encoding(s) used for missing values : {encodings_str}")
        print(f"Total number of affected rows     : {total_affected_rows}")
        print(f"Columns containing them           : {affected_cols_str}")
    else:
        print("No non-standard encodings to summarize.")

if __name__ == '__main__':
    validate_schema('adult_income.csv')
