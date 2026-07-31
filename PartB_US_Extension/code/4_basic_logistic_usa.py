# 4_basic_logistic_usa.py

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


# Load preprocessed US data
df = pd.read_csv(DATA_DIR / "cleaned_data_preprocessing_usa.csv")

# Outcome variable: high mask-wearing
df["mask_binary"] = df["face_mask_behaviour_binary"].map({
    "No": 0,
    "Yes": 1
})

# Basic numerical control variables
base_predictors = [
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

# Keep only variables that exist
base_predictors = [col for col in base_predictors if col in df.columns]

# Dummy-variable controls already created in 02_data_preprocessing_usa.py.
# not the state fixed-effects model.
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

predictors = base_predictors + dummy_predictors

# Build modelling dataset
model_df = df[["mask_binary"] + predictors].copy()

# Convert all variables to numeric
model_df["mask_binary"] = pd.to_numeric(model_df["mask_binary"], errors="coerce")

for col in predictors:
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

# Remove rows with missing model variables
model_df = model_df.dropna()

y = model_df["mask_binary"].astype(float)
X = model_df[predictors].astype(float)

# Add intercept
X = sm.add_constant(X, has_constant="add")

# Fit logistic regression
model = sm.Logit(y, X)
result = model.fit(maxiter=200)

print(result.summary())

# Create odds-ratio table
conf = result.conf_int()
results_table = pd.DataFrame({
    "variable": result.params.index,
    "coef": result.params.values,
    "odds_ratio": [safe_exp(x) for x in result.params.values],
    "ci_lower": [safe_exp(x) for x in conf[0].values],
    "ci_upper": [safe_exp(x) for x in conf[1].values],
    "p_value": result.pvalues.values
})

# Save results
results_table.to_csv(RESULTS_DIR / "04_basic_logistic_usa_results.csv", index=False)

# Print key result
print("\n===== Key Result: within_mandate_period =====")
print(
    results_table[
        results_table["variable"] == "within_mandate_period"
    ]
)

print("\nModel sample size:", model_df.shape[0])
print("Number of predictors:", len(predictors))
print("Saved to:", RESULTS_DIR / "04_basic_logistic_usa_results.csv")
