# 00_mask_mandates_usa.py

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw_data"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Load US policy data
df = pd.read_csv(RAW_DIR / "OxCGRT_US_latest.csv")

# Select columns
col_subsets = ["RegionName", "RegionCode", "Date", "H6M_Facial Coverings"]

df.index = pd.to_datetime(df["Date"], format="%Y%m%d")
df = df.loc[:, col_subsets]

# Find rolling averages
rolling_days = 14
df_rolling = df.loc[:, ["RegionName", "H6M_Facial Coverings"]].groupby(
    "RegionName"
).rolling(window=rolling_days).mean()

# Find first time mandates are consistently put in place
mandate_limit = 3
df_mandates = df_rolling[
    df_rolling["H6M_Facial Coverings"] >= mandate_limit
].groupby("RegionName").head(1)

# Check result
print(df_mandates)

# Save data
df_mandates.to_csv(DATA_DIR / "us_mandate_start_dates.csv")
print("Saved to:", DATA_DIR / "us_mandate_start_dates.csv")
