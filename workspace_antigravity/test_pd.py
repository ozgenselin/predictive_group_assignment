import pandas as pd
try:
    df = pd.read_csv('adult_income.csv')
    for col in df.columns:
        print(f"{col}: {df[col].dtype}, {df[col].dtype.name}")
except Exception as e:
    print(f"Error: {e}")
