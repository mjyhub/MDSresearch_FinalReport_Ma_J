# 4b_rq2_holdout_robustness_usa.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

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


def fit_logit(sample_df, predictors, sample_name):
    y = sample_df["mask_binary"].astype(float)
    X = sample_df[predictors].astype(float)

    X = sm.add_constant(X, has_constant="add")

    model = sm.Logit(y, X)
    result = model.fit(maxiter=200, disp=False)

    conf = result.conf_int()

    results_table = pd.DataFrame({
        "sample": sample_name,
        "variable": result.params.index,
        "coef": result.params.values,
        "odds_ratio": [safe_exp(x) for x in result.params.values],
        "ci_lower": [safe_exp(x) for x in conf[0].values],
        "ci_upper": [safe_exp(x) for x in conf[1].values],
        "p_value": result.pvalues.values,
        "n": sample_df.shape[0]
    })

    return result, results_table


# -----------------------------
# Load and prepare data
# -----------------------------

df = pd.read_csv(DATA_DIR / "cleaned_data_preprocessing_usa.csv")

# Outcome variable: high mask-wearing
df["mask_binary"] = df["face_mask_behaviour_binary"].map({
    "No": 0,
    "Yes": 1
})

# Same predictors as 4_basic_logistic_usa.py
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

base_predictors = [col for col in base_predictors if col in df.columns]

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

# -----------------------------
# 80/20 hold-out split
# -----------------------------

train_df, test_df = train_test_split(
    model_df,
    test_size=0.2,
    random_state=20240417,
    stratify=model_df["mask_binary"]
)

# -----------------------------
# Fit same model in full/train/test samples
# -----------------------------

full_result, full_table = fit_logit(
    model_df,
    predictors,
    "full_sample"
)

train_result, train_table = fit_logit(
    train_df,
    predictors,
    "training_sample"
)

test_result, test_table = fit_logit(
    test_df,
    predictors,
    "testing_sample"
)

all_results = pd.concat(
    [full_table, train_table, test_table],
    ignore_index=True
)

# Save all coefficients
all_results.to_csv(
    RESULTS_DIR / "04b_rq2_holdout_all_coefficients.csv",
    index=False
)

# Save key result only
key_results = all_results[
    all_results["variable"] == "within_mandate_period"
].copy()

key_results.to_csv(
    RESULTS_DIR / "04b_rq2_holdout_within_mandate_results.csv",
    index=False
)

# -----------------------------
# Evaluate training model on held-out testing sample
# -----------------------------

X_test = test_df[predictors].astype(float)
X_test = sm.add_constant(X_test, has_constant="add")
y_test = test_df["mask_binary"].astype(float)

test_prob = train_result.predict(X_test)
test_pred = (test_prob >= 0.5).astype(int)

test_auc = roc_auc_score(y_test, test_prob)
test_accuracy = accuracy_score(y_test, test_pred)

performance_table = pd.DataFrame({
    "metric": [
        "test_auc",
        "test_accuracy",
        "training_n",
        "testing_n",
        "full_n"
    ],
    "value": [
        test_auc,
        test_accuracy,
        train_df.shape[0],
        test_df.shape[0],
        model_df.shape[0]
    ]
})

performance_table.to_csv(
    RESULTS_DIR / "04b_rq2_holdout_test_performance.csv",
    index=False
)

# -----------------------------
# Print summary
# -----------------------------

print("\n===== RQ2 Hold-out Robustness: Key Result =====")
print(key_results)

print("\n===== Held-out Test Performance =====")
print(performance_table)

print("\nSaved all coefficients to:")
print(RESULTS_DIR / "04b_rq2_holdout_all_coefficients.csv")

print("\nSaved key mandate-period results to:")
print(RESULTS_DIR / "04b_rq2_holdout_within_mandate_results.csv")

print("\nSaved test performance to:")
print(RESULTS_DIR / "04b_rq2_holdout_test_performance.csv")