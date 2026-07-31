# 6b_rq3_holdout_robustness_interaction.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

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

    zero_variance_cols = [
        col for col in X.columns
        if X[col].nunique() <= 1
    ]

    if zero_variance_cols:
        print(f"\n{sample_name}: dropping zero-variance columns:")
        print(zero_variance_cols)
        X = X.drop(columns=zero_variance_cols)

    used_predictors = X.columns.tolist()

    X = sm.add_constant(X, has_constant="add")

    model = sm.Logit(y, X)
    result = model.fit(maxiter=300, disp=False)

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

    beta_mandate_aus = result.params.get("within_mandate_period", np.nan)
    beta_interaction = result.params.get("mandate_x_US", np.nan)

    helper_table = pd.DataFrame({
        "sample": [sample_name, sample_name, sample_name],
        "quantity": [
            "Australia mandate OR",
            "US vs Australia interaction OR",
            "US implied mandate OR"
        ],
        "value": [
            safe_exp(beta_mandate_aus),
            safe_exp(beta_interaction),
            safe_exp(beta_mandate_aus + beta_interaction)
        ],
        "n": [sample_df.shape[0]] * 3
    })

    return result, results_table, helper_table, used_predictors


# -----------------------------
# Load combined Australia-US data
# -----------------------------

df = pd.read_csv(DATA_DIR / "combined_aus_us_for_interaction.csv")

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

model_df = df[["mask_binary", "country"] + predictors].copy()

model_df["mask_binary"] = pd.to_numeric(
    model_df["mask_binary"],
    errors="coerce"
)

for col in predictors:
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

model_df = model_df.dropna()

# Stratify by country and outcome so both samples keep similar Australia/US and outcome balance
model_df["split_stratum"] = (
    model_df["country"].astype(str)
    + "_"
    + model_df["mask_binary"].astype(int).astype(str)
)

train_df, test_df = train_test_split(
    model_df,
    test_size=0.2,
    random_state=20240417,
    stratify=model_df["split_stratum"]
)

train_df = train_df.drop(columns=["split_stratum"])
test_df = test_df.drop(columns=["split_stratum"])
model_df = model_df.drop(columns=["split_stratum"])

# -----------------------------
# Fit full/train/test interaction models
# -----------------------------

full_result, full_table, full_helper, full_predictors = fit_logit(
    model_df,
    predictors,
    "full_sample"
)

train_result, train_table, train_helper, train_predictors = fit_logit(
    train_df,
    predictors,
    "training_sample"
)

test_result, test_table, test_helper, test_predictors = fit_logit(
    test_df,
    predictors,
    "testing_sample"
)

all_results = pd.concat(
    [full_table, train_table, test_table],
    ignore_index=True
)

helper_results = pd.concat(
    [full_helper, train_helper, test_helper],
    ignore_index=True
)

all_results.to_csv(
    RESULTS_DIR / "06b_rq3_holdout_all_coefficients.csv",
    index=False
)

key_vars = [
    "within_mandate_period",
    "country_US",
    "mandate_x_US"
]

key_results = all_results[
    all_results["variable"].isin(key_vars)
].copy()

key_results.to_csv(
    RESULTS_DIR / "06b_rq3_holdout_key_results.csv",
    index=False
)

helper_results.to_csv(
    RESULTS_DIR / "06b_rq3_holdout_interpretation_helpers.csv",
    index=False
)

# -----------------------------
# Predict testing set using training model
# -----------------------------

X_test = test_df[train_predictors].astype(float)
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
    RESULTS_DIR / "06b_rq3_holdout_test_performance.csv",
    index=False
)

print("\n===== RQ3 Hold-out Robustness: Key Results =====")
print(key_results)

print("\n===== Interpretation Helpers =====")
print(helper_results)

print("\n===== Held-out Test Performance =====")
print(performance_table)