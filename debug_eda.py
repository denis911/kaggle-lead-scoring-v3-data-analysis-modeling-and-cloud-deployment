import pandas as pd
import numpy as np

# 1. Load Data
print("Loading data...")
df = pd.read_csv('Xyz.csv', encoding='latin1')

# 2. Initial Shape
print(f"Initial Shape: {df.shape}")

# 3. Clean 'Select' values
# Replace 'Select' with NaN
df = df.replace('Select', np.nan)

# 4. Disqualification Logic (Non-Diabetic)
# Check columns for 'Diabetes' related status
# Based on PRELIMINARY analysis: "Non Diabetic" status leads to disqualification (Dead Lead)
# We need to find the specific column. Analysis said "Diabetes Status" or "Clinical"
# Let's inspect columns to find the relevant one
print("\nScanning columns for 'Diabetes' or 'Status'...")
possible_cols = [c for c in df.columns if 'Diabetes' in c or 'Status' in c or 'Disposition' in c]
print(f"Possible Status Columns: {possible_cols}")

# 5. Drop High Missingness Columns (>40%)
missing_series = df.isnull().mean()
drop_cols = missing_series[missing_series > 0.4].index.tolist()
print(f"\nDropping {len(drop_cols)} columns with >40% missing values: {drop_cols}")
df_dropped = df.drop(columns=drop_cols)
print(f"Shape after dropping missing: {df_dropped.shape}")

# 6. Target Variable Analysis
# Target is 'Converted' (binary) if it exists, or 'Lead Stage'
if 'Converted' in df.columns:
    print("\nTarget 'Converted' breakdown:")
    print(df['Converted'].value_counts(normalize=True))
elif 'Lead Stage' in df.columns:
    print("\nTarget 'Lead Stage' breakdown:")
    print(df['Lead Stage'].value_counts(normalize=True))

