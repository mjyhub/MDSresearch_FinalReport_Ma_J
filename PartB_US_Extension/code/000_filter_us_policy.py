# 000_filter_us_policy.py

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw_data"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Load OxCGRT subnational data
df = pd.read_csv(RAW_DIR / "OxCGRT_compact_subnational_v1.csv", low_memory=False)

# Filter United States data
df_us = df[df["CountryCode"] == "USA"].copy()

# Keep state-level total policy environment
df_us_state = df_us[df_us["Jurisdiction"] == "STATE_TOTAL"].copy()

# Keep relevant columns
col_subsets = [
    "CountryName",
    "CountryCode",
    "RegionName",
    "RegionCode",
    "Jurisdiction",
    "Date",
    "H6M_Facial.Coverings"
]

df_us_state = df_us_state.loc[:, col_subsets]
df_us_state = df_us_state.rename(columns={
    "H6M_Facial.Coverings": "H6M_Facial Coverings"
})

# Check result
print(df_us_state.shape)
print(df_us_state.head())
print("Number of regions:", df_us_state["RegionName"].nunique())
print(sorted(df_us_state["RegionName"].dropna().unique()))

# Save US policy data
df_us_state.to_csv(RAW_DIR / "OxCGRT_US_latest.csv", index=False)
print("Saved to:", RAW_DIR / "OxCGRT_US_latest.csv")
