# 0_missing_table_usa.py

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw_data"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Load US YouGov data
df = pd.read_csv(
    RAW_DIR / "united-states.csv",
    na_values=[" ", "", "__NA__"],
    keep_default_na=True,
    low_memory=False
)

# Count missing values for each variable
missing_value_counts = {}

for col in df:
    missing_count = df[col].isna().sum()
    missing_value_counts[col] = missing_count

# Convert to DataFrame
missing_value_df = pd.DataFrame(
    list(missing_value_counts.items()),
    columns=["Variable Name", "Missing Value Count"]
)

# Add missing percentage
missing_value_df["Missing Value Percent"] = (
    missing_value_df["Missing Value Count"] / len(df) * 100
)

# Sort by missing value count and variable name
missing_value_df = missing_value_df.sort_values(
    by=["Missing Value Count", "Variable Name"]
)

# Save result
missing_value_df.to_csv(DATA_DIR / "missing_value_counts_usa.csv", index=False)

# Print basic summary
print("Number of rows:", len(df))
print("Number of columns:", df.shape[1])
print(missing_value_df.head(20))
print(missing_value_df.tail(20))
print("Saved to:", DATA_DIR / "missing_value_counts_usa.csv")
