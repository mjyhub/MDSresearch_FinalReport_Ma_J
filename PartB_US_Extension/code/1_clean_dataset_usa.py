# 1_clean_dataset_usa.py

import pandas as pd
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw_data"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def convert_datetime(dt):
    date = str(dt).split()[0]
    return datetime.strptime(date, "%d/%m/%Y")


def household_convert(size_str):
    size_str = str(size_str)

    for i in range(1, 8):
        if size_str == str(i):
            return i

    if size_str == "8 or more":
        return 8

    if size_str in ["Prefer not to say", "Don't know", "nan"]:
        return None

    return None


# Load US YouGov data
df = pd.read_csv(
    RAW_DIR / "united-states.csv",
    na_values=[" ", "", "__NA__"],
    keep_default_na=True,
    low_memory=False
)

# Convert endtime to date
df["endtime"] = df["endtime"].apply(convert_datetime)

# Load missing-value table
missing_value_df = pd.read_csv(DATA_DIR / "missing_value_counts_usa.csv")

# Use 20% missingness threshold, following the Part A logic
thresh_value = 0.2 * len(df)

# Variables important for comparability with Part A
core_cols = [
    "RecordNo",
    "endtime",
    "state",
    "qweek",
    "weight",
    "gender",
    "age",
    "household_size",
    "household_children",
    "employment_status",
    "i12_health_1",
    "i12_health_22",
    "i12_health_23",
    "i12_health_25",
    "i2_health",
    "i7a_health",
    "i9_health",
    "i10_health",
    "i11_health",
    "i13_health",
    "WCRex1",
    "WCRex2",
    "cantril_ladder",
    "PHQ4_1",
    "PHQ4_2",
    "PHQ4_3",
    "PHQ4_4"
]

# Drop highly missing variables, but keep important comparable variables
columns_to_drop = missing_value_df.loc[
    missing_value_df["Missing Value Count"] > thresh_value,
    "Variable Name"
].tolist()

columns_to_drop = [col for col in columns_to_drop if col not in core_cols]

df.drop(columns=columns_to_drop, inplace=True, errors="ignore")

# Convert frequency responses to numerical scale
frequency_dict = {
    "Always": 5,
    "Frequently": 4,
    "Sometimes": 3,
    "Rarely": 2,
    "Not at all": 1
}

for column in df.columns:
    if column.startswith("i12_health_"):
        df[column] = df[column].map(frequency_dict)

# Create face mask behaviour scale
face_mask_cols = [
    "i12_health_1",
    "i12_health_22",
    "i12_health_23",
    "i12_health_25"
]

face_mask_cols = [col for col in face_mask_cols if col in df.columns]

df["face_mask_behaviour_scale"] = df[face_mask_cols].median(
    axis=1,
    skipna=True
)

df["face_mask_behaviour_binary"] = df["face_mask_behaviour_scale"].apply(
    lambda x: "Yes" if x >= 4 else "No"
)

# Create general protective behaviour scale
protective_behaviour_cols = [
    col for col in df.columns
    if col.startswith("i12_health_")
]

df["protective_behaviour_scale"] = df[protective_behaviour_cols].median(
    axis=1,
    skipna=True
)

df["protective_behaviour_binary"] = df["protective_behaviour_scale"].apply(
    lambda x: "Yes" if x >= 4 else "No"
)

# Protective behaviour excluding face mask items
protective_behaviour_nomask_cols = [
    col for col in protective_behaviour_cols
    if col not in face_mask_cols
]

df["protective_behaviour_nomask_scale"] = df[
    protective_behaviour_nomask_cols
].median(axis=1, skipna=True)

# Recreate week number using fortnight intervals
start_date = df["endtime"].min()
df["week_number"] = ((df["endtime"] - start_date).dt.days // 14) + 1

# Convert household size to numeric
df["household_size"] = df["household_size"].apply(household_convert)

# Remove rows missing key modelling variables
key_required_cols = [
    "state",
    "endtime",
    "face_mask_behaviour_scale",
    "face_mask_behaviour_binary",
    "protective_behaviour_nomask_scale",
    "gender",
    "age",
    "household_size",
    "employment_status"
]

key_required_cols = [col for col in key_required_cols if col in df.columns]

df = df.dropna(subset=key_required_cols)

# Drop qweek, weight, and original protective behaviour items
drop_cols = ["qweek", "weight"] + protective_behaviour_cols
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

# Save cleaned US data
df.to_csv(DATA_DIR / "cleaned_data_usa.csv", index=False)

# Checks
print("Cleaned shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFace mask binary distribution:")
print(df["face_mask_behaviour_binary"].value_counts(dropna=False))
print(df["face_mask_behaviour_binary"].value_counts(normalize=True, dropna=False))

print("\nProtective behaviour binary distribution:")
print(df["protective_behaviour_binary"].value_counts(dropna=False))
print(df["protective_behaviour_binary"].value_counts(normalize=True, dropna=False))

print("\nNumber of states:")
print(df["state"].nunique())

print("\nSaved to:")
print(DATA_DIR / "cleaned_data_usa.csv")