# 2_state_level_descriptive_comparison.py

import pandas as pd
import matplotlib.pyplot as plt
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


# -----------------------------
# Load cleaned US data
# -----------------------------

df = pd.read_csv(
    DATA_DIR / "cleaned_data_usa.csv",
    keep_default_na=False,
    low_memory=False
)

df["endtime"] = pd.to_datetime(df["endtime"])

# Convert outcome to 0/1
df["mask_binary"] = df["face_mask_behaviour_binary"].map({
    "No": 0,
    "Yes": 1
})

# Harmonise state name for policy merge
df["state_merge"] = df["state"].replace({
    "District of Columbia": "Washington DC"
})

# -----------------------------
# Load mandate start dates
# -----------------------------

mandate_df = pd.read_csv(DATA_DIR / "us_mandate_start_dates.csv")
mandate_df["mandate_start_date"] = pd.to_datetime(mandate_df["Date"])
mandate_df = mandate_df[["RegionName", "mandate_start_date"]].copy()

# Merge mandate start dates
df = df.merge(
    mandate_df,
    left_on="state_merge",
    right_on="RegionName",
    how="left"
)

# States without sustained mandate start date are coded as never in mandate period
states_without_mandate = sorted(
    df.loc[df["mandate_start_date"].isna(), "state"].dropna().unique()
)

print("States without sustained mandate start date:")
print(states_without_mandate)

# Create within_mandate_period
df["within_mandate_period"] = df.apply(mandates_convert, axis=1)

# -----------------------------
# State-level observed proportions
# -----------------------------

state_period_summary = (
    df.groupby(["state", "within_mandate_period"])
    .agg(
        n=("mask_binary", "size"),
        high_mask_n=("mask_binary", "sum"),
        high_mask_prop=("mask_binary", "mean")
    )
    .reset_index()
)

state_period_summary["period_label"] = state_period_summary[
    "within_mandate_period"
].map({
    0: "Before/no sustained mandate",
    1: "During sustained mandate"
})

state_period_summary.to_csv(
    RESULTS_DIR / "02_state_mask_wearing_by_period.csv",
    index=False
)

# -----------------------------
# Pivot to compare before and during mandate periods
# -----------------------------

state_diff = state_period_summary.pivot(
    index="state",
    columns="within_mandate_period",
    values="high_mask_prop"
).reset_index()

state_diff = state_diff.rename(columns={
    0: "before_or_no_mandate_prop",
    1: "during_mandate_prop"
})

# Add counts by period
state_counts = state_period_summary.pivot(
    index="state",
    columns="within_mandate_period",
    values="n"
).reset_index()

state_counts = state_counts.rename(columns={
    0: "before_or_no_mandate_n",
    1: "during_mandate_n"
})

state_diff = state_diff.merge(state_counts, on="state", how="left")

# Keep only states with a sustained mandate period for the before/during comparison
state_diff["has_sustained_mandate"] = ~state_diff["during_mandate_prop"].isna()

# Proportional change:
# 100 x (Y_during - Y_before) / Y_before
state_diff["proportional_change_pct"] = (
    (
        state_diff["during_mandate_prop"] -
        state_diff["before_or_no_mandate_prop"]
    )
    / state_diff["before_or_no_mandate_prop"]
) * 100

state_diff = state_diff.sort_values("proportional_change_pct", ascending=True)

state_diff.to_csv(
    RESULTS_DIR / "02_state_mask_wearing_difference.csv",
    index=False
)

# -----------------------------
# Plot state-level proportional changes
# -----------------------------

plot_df = state_diff[state_diff["has_sustained_mandate"]].copy()

plt.figure(figsize=(10, 12))

colors = [
    "#2C7FB8" if value >= 0 else "#D95F0E"
    for value in plot_df["proportional_change_pct"]
]

plt.barh(
    plot_df["state"],
    plot_df["proportional_change_pct"],
    color=colors
)

plt.axvline(0, color="black", linewidth=0.8)

plt.xlabel(
    "Proportional change in high mask-wearing (%)\n"
)
plt.ylabel("State")


plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "02_state_level_descriptive_comparison.png",
    dpi=300
)

plt.close()

# -----------------------------
# Print checks
# -----------------------------

print("\nState-period summary saved to:")
print(RESULTS_DIR / "02_state_mask_wearing_by_period.csv")

print("\nState proportional change summary saved to:")
print(RESULTS_DIR / "02_state_mask_wearing_difference.csv")

print("\nPlot saved to:")
print(RESULTS_DIR / "02_state_level_descriptive_comparison.png")

print("\nTop 10 largest proportional increases:")
print(
    state_diff.sort_values("proportional_change_pct", ascending=False)
    .head(10)[[
        "state",
        "before_or_no_mandate_prop",
        "during_mandate_prop",
        "proportional_change_pct",
        "before_or_no_mandate_n",
        "during_mandate_n"
    ]]
)

print("\nTop 10 smallest proportional increases or decreases:")
print(
    state_diff.sort_values("proportional_change_pct", ascending=True)
    .head(10)[[
        "state",
        "before_or_no_mandate_prop",
        "during_mandate_prop",
        "proportional_change_pct",
        "before_or_no_mandate_n",
        "during_mandate_n"
    ]]
)

print("\nStates without sustained mandate start date:")
print(states_without_mandate)