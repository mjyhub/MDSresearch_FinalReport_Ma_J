# 3_data_preprocessing_usa.py

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw_data"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def mandates_convert(row):
    endtime = pd.to_datetime(row["endtime"])

    if pd.isna(row["mandate_start_date"]):
        return 0

    if row["mandate_start_date"] <= endtime:
        return 1
    else:
        return 0


# Load cleaned US YouGov data
cleaned_df = pd.read_csv(
    DATA_DIR / "cleaned_data_usa.csv",
    keep_default_na=False
)

# Make sure endtime is datetime
cleaned_df["endtime"] = pd.to_datetime(cleaned_df["endtime"])

# Harmonise state names before merging
cleaned_df["state_merge"] = cleaned_df["state"].replace({
    "District of Columbia": "Washington DC"
})

# Load US mandate start dates
mandate_df = pd.read_csv(DATA_DIR / "us_mandate_start_dates.csv")

# Make sure mandate date is datetime
mandate_df["mandate_start_date"] = pd.to_datetime(mandate_df["Date"])

# Keep only the useful columns
mandate_df = mandate_df[["RegionName", "mandate_start_date"]].copy()

# Merge mandate dates into individual-level YouGov data
cleaned_df = cleaned_df.merge(
    mandate_df,
    left_on="state_merge",
    right_on="RegionName",
    how="left"
)

# Check states without sustained mandate start date
states_without_mandate_start = cleaned_df.loc[
    cleaned_df["mandate_start_date"].isna(),
    "state"
].unique()

print("States without sustained mandate start date:")
print(states_without_mandate_start)

# Create within_mandate_period
cleaned_df["within_mandate_period"] = cleaned_df.apply(
    mandates_convert,
    axis=1
)

# Convert to integer
cleaned_df["within_mandate_period"] = cleaned_df["within_mandate_period"].astype(int)

# Drop merge helper columns
cleaned_df = cleaned_df.drop(
    columns=["RegionName", "mandate_start_date", "state_merge"],
    errors="ignore"
)

# Convert categorical variables into dummy variables
convert_into_dummy_cols = [
    "state",
    "gender",
    "i9_health",
    "employment_status",
    "i11_health",
    "WCRex1",
    "WCRex2",
    "PHQ4_1",
    "PHQ4_2",
    "PHQ4_3",
    "PHQ4_4"
]

# Only use columns that exist in cleaned_df
convert_into_dummy_cols = [
    col for col in convert_into_dummy_cols
    if col in cleaned_df.columns
]

for col in convert_into_dummy_cols:
    dummy = pd.get_dummies(cleaned_df[col], prefix=col, drop_first=True)
    cleaned_df = pd.concat([cleaned_df, dummy], axis=1)
    cleaned_df = cleaned_df.drop(col, axis=1)

# Convert dummy boolean columns to integer 0/1
bool_cols = cleaned_df.select_dtypes(include=["bool"]).columns

for col in bool_cols:
    cleaned_df[col] = cleaned_df[col].astype(int)

# Check whether any boolean columns remain
remaining_bool_cols = cleaned_df.select_dtypes(include=["bool"]).columns.tolist()

print("\nRemaining boolean columns:")
print(remaining_bool_cols)

# Check dummy columns contain only 0/1
dummy_prefixes = [
    "state_",
    "gender_",
    "i9_health_",
    "employment_status_",
    "i11_health_",
    "WCRex1_",
    "WCRex2_",
    "PHQ4_1_",
    "PHQ4_2_",
    "PHQ4_3_",
    "PHQ4_4_"
]

dummy_cols = [
    col for col in cleaned_df.columns
    if any(col.startswith(prefix) for prefix in dummy_prefixes)
]

invalid_dummy_cols = []

for col in dummy_cols:
    unique_values = set(cleaned_df[col].dropna().unique())
    if not unique_values.issubset({0, 1}):
        invalid_dummy_cols.append((col, unique_values))

print("\nNumber of dummy columns:", len(dummy_cols))
print("Invalid dummy columns:")
print(invalid_dummy_cols[:20])

# Save preprocessing data
cleaned_df.to_csv(DATA_DIR / "cleaned_data_preprocessing_usa.csv", index=False)

# Check result
print("Preprocessed shape:")
print(cleaned_df.shape)

print("\nWithin mandate period distribution:")
print(cleaned_df["within_mandate_period"].value_counts())
print(cleaned_df["within_mandate_period"].value_counts(normalize=True))
print("Saved to:", DATA_DIR / "cleaned_data_preprocessing_usa.csv")
