# 6_country_interaction_model.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw_data"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def safe_exp(x):
    if x > 700:
        return np.inf
    if x < -700:
        return 0.0
    return np.exp(x)


# Load combined Australia-US dataset
df = pd.read_csv(DATA_DIR / "combined_aus_us_for_interaction.csv")

# Core predictors for country interaction model
core_predictors = [
    "within_mandate_period",
    "country_US",
    "mandate_x_US",
    "age",
    "household_size",
    "i2_health",
    "protective_behaviour_nomask_scale",
    "week_number"
]

core_predictors = [col for col in core_predictors if col in df.columns]

# Comparable dummy controls retained in 05_prepare_combined_aus_us.py
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

dummy_predictors = [
    col for col in df.columns
    if any(col.startswith(prefix) for prefix in dummy_prefixes)
]

predictors = core_predictors + dummy_predictors

# Build modelling dataset
model_df = df[["mask_binary", "country"] + predictors].copy()

# Convert all model variables except country to numeric
model_df["mask_binary"] = pd.to_numeric(model_df["mask_binary"], errors="coerce")

for col in predictors:
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

# Remove rows with missing model variables
model_df = model_df.dropna()

y = model_df["mask_binary"].astype(float)
X = model_df[predictors].astype(float)

# Drop zero-variance predictors if any
zero_variance_cols = [
    col for col in X.columns
    if X[col].nunique() <= 1
]

if zero_variance_cols:
    print("Dropping zero-variance columns:")
    print(zero_variance_cols)
    X = X.drop(columns=zero_variance_cols)

# Add intercept
X = sm.add_constant(X, has_constant="add")

print("Model sample size:", X.shape[0])
print("Number of predictors:", X.shape[1] - 1)

print("\nCountry distribution in model sample:")
print(model_df["country"].value_counts())

print("\nMandate period by country in model sample:")
print(pd.crosstab(
    model_df["country"],
    model_df["within_mandate_period"],
    margins=True
))

# Fit logistic regression
model = sm.Logit(y, X)
result = model.fit(maxiter=300)

print(result.summary())

# Create results table
conf = result.conf_int()

results_table = pd.DataFrame({
    "variable": result.params.index,
    "coef": result.params.values,
    "odds_ratio": [safe_exp(x) for x in result.params.values],
    "ci_lower": [safe_exp(x) for x in conf[0].values],
    "ci_upper": [safe_exp(x) for x in conf[1].values],
    "p_value": result.pvalues.values
})

# Save full results
results_table.to_csv(
    RESULTS_DIR / "06_country_interaction_model_results.csv",
    index=False
)

# Save key results only
key_vars = [
    "within_mandate_period",
    "country_US",
    "mandate_x_US"
]

key_results = results_table[
    results_table["variable"].isin(key_vars)
].copy()

key_results.to_csv(
    RESULTS_DIR / "06_country_interaction_key_results.csv",
    index=False
)

print("\n===== Key Results =====")
print(key_results)

# Helpful interpretation numbers
beta_mandate_aus = result.params.get("within_mandate_period", np.nan)
beta_interaction = result.params.get("mandate_x_US", np.nan)

or_mandate_aus = safe_exp(beta_mandate_aus)
or_interaction = safe_exp(beta_interaction)
or_mandate_us = safe_exp(beta_mandate_aus + beta_interaction)

print("\n===== Interpretation Helpers =====")
print("Australia mandate odds ratio:", or_mandate_aus)
print("US vs Australia interaction odds ratio:", or_interaction)
print("US mandate odds ratio implied by model:", or_mandate_us)

print("\nSaved full results to:")
print(RESULTS_DIR / "06_country_interaction_model_results.csv")

print("\nSaved key results to:")
print(RESULTS_DIR / "06_country_interaction_key_results.csv")