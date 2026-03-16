import pandas as pd
import numpy as np

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
        dtype = str(df[col].dtype)
        missing_count = standard_missing[col]
        # Calculate unique counts only for categorical (object or category)
        if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col].dtype):
            unique_count = str(df[col].nunique())
        else:
            unique_count = "N/A"
            
        print(f"{col:<20} | {dtype:<10} | {missing_count:<18} | {unique_count}")

    print("\n" + "="*80)
    print(" NON-STANDARD MISSING VALUES DETECTION")
    print("="*80)
    
    # Non-standard missing values are often strings like '?', ' ?', 'Unknown', etc.
    # We will look for strings containing '?' or empty strings.
    non_standard_missing = {}
    affected_rows_idx = set()
    
    # We will iterate through object columns to find suspect values
    for col in df.columns:
        if df[col].dtype == 'object':
            # Get value counts to inspect
            val_counts = df[col].value_counts(dropna=False)
            for val, count in val_counts.items():
                if isinstance(val, str):
                    stripped_val = val.strip()
                    # Check for common placeholder values
                    if stripped_val in ['?', 'Unknown', 'unknown', 'N/A', 'n/a', 'NA', 'na', '', '-']:
                        if col not in non_standard_missing:
                            non_standard_missing[col] = {}
                        non_standard_missing[col][val] = count
                        
                        # Collect indices for affected rows
                        idx = df[df[col] == val].index
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
    # No hardcoded absolute path, use relative 'adult_income.csv'
    validate_schema('adult_income.csv')
