# 5_prepare_combined_aus_us.py

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw_data"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# -----------------------------
# Load Australia and US data
# -----------------------------

aus_df = pd.read_csv(
    DATA_DIR / "aus_cleaned_data_preprocessing.csv",
    keep_default_na=False,
    low_memory=False
)

us_df = pd.read_csv(
    DATA_DIR / "cleaned_data_preprocessing_usa.csv",
    keep_default_na=False,
    low_memory=False
)

# -----------------------------
# Add country labels
# -----------------------------

aus_df["country"] = "Australia"
us_df["country"] = "US"

# -----------------------------
# Convert outcome to 0/1
# -----------------------------

aus_df["mask_binary"] = aus_df["face_mask_behaviour_binary"].map({
    "No": 0,
    "Yes": 1
})

us_df["mask_binary"] = us_df["face_mask_behaviour_binary"].map({
    "No": 0,
    "Yes": 1
})

# -----------------------------
# Make sure mandate variable is numeric
# -----------------------------

aus_df["within_mandate_period"] = pd.to_numeric(
    aus_df["within_mandate_period"],
    errors="coerce"
)

us_df["within_mandate_period"] = pd.to_numeric(
    us_df["within_mandate_period"],
    errors="coerce"
)

# -----------------------------
# Select variables that should exist in both datasets
# -----------------------------

required_cols = [
    "country",
    "mask_binary",
    "within_mandate_period",
    "age",
    "household_size",
    "household_children",
    "i2_health",
    "i7a_health",
    "i13_health",
    "protective_behaviour_nomask_scale",
    "week_number"
]

# Keep only required columns that exist in both datasets
common_required_cols = [
    col for col in required_cols
    if col in aus_df.columns and col in us_df.columns
]

print("Common required columns:")
print(common_required_cols)

# -----------------------------
# Add comparable dummy controls
# -----------------------------
# These prefixes are created in preprocessing for both countries.
# We only keep dummy columns that exist in both datasets.

dummy_prefixes = [
    "gender_",
    "employment_status_",
    "i9_health_",
    "i11_health_",
    "WCRex1_",
    "WCRex2_",
    "PHQ4_1_",
    "PHQ4_2_",
    "PHQ4_3_",
    "PHQ4_4_"
]

aus_dummy_cols = [
    col for col in aus_df.columns
    if any(col.startswith(prefix) for prefix in dummy_prefixes)
]

us_dummy_cols = [
    col for col in us_df.columns
    if any(col.startswith(prefix) for prefix in dummy_prefixes)
]

common_dummy_cols = sorted(list(set(aus_dummy_cols).intersection(set(us_dummy_cols))))

print("\nNumber of common dummy columns:")
print(len(common_dummy_cols))

print("\nCommon dummy columns:")
print(common_dummy_cols)

# -----------------------------
# Build combined dataset
# -----------------------------

combined_cols = common_required_cols + common_dummy_cols

aus_model = aus_df[combined_cols].copy()
us_model = us_df[combined_cols].copy()

combined_df = pd.concat(
    [aus_model, us_model],
    axis=0,
    ignore_index=True
)

# -----------------------------
# Create country indicator and interaction term
# -----------------------------

combined_df["country_US"] = (combined_df["country"] == "US").astype(int)

combined_df["mandate_x_US"] = (
    combined_df["within_mandate_period"] * combined_df["country_US"]
)

# -----------------------------
# Convert all model columns except country to numeric
# -----------------------------

for col in combined_df.columns:
    if col != "country":
        combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")

# Remove rows with missing core variables
core_model_cols = [
    "mask_binary",
    "within_mandate_period",
    "country_US",
    "mandate_x_US"
]

combined_df = combined_df.dropna(subset=core_model_cols)

# -----------------------------
# Save combined dataset
# -----------------------------

combined_df.to_csv(DATA_DIR / "combined_aus_us_for_interaction.csv", index=False)

# -----------------------------
# Checks
# -----------------------------

print("\nCombined dataset shape:")
print(combined_df.shape)

print("\nCountry distribution:")
print(combined_df["country"].value_counts())

print("\nMandate period by country:")
print(pd.crosstab(
    combined_df["country"],
    combined_df["within_mandate_period"],
    margins=True
))

print("\nMask-wearing outcome by country:")
print(pd.crosstab(
    combined_df["country"],
    combined_df["mask_binary"],
    normalize="index"
))

print("\nMissing values in core columns:")
print(combined_df[core_model_cols].isna().sum())

print("\nSaved to:")
print(DATA_DIR / "combined_aus_us_for_interaction.csv")
