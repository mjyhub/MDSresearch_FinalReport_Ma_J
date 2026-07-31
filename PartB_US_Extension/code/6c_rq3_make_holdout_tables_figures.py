# 6c_rq3_make_holdout_tables_figures.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"


def format_p_value(p):
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def format_n(n):
    return f"{int(n):,}"


# -----------------------------
# 1. Read RQ3 hold-out key results
# -----------------------------

key_df = pd.read_csv(
    RESULTS_DIR / "06b_rq3_holdout_key_results.csv"
)

sample_labels = {
    "full_sample": "Full sample",
    "training_sample": "Training sample",
    "testing_sample": "Testing sample"
}

variable_labels = {
    "within_mandate_period": "Mandate period",
    "country_US": "US country indicator",
    "mandate_x_US": "Mandate period x US"
}

key_df["Sample"] = key_df["sample"].map(sample_labels)
key_df["Term"] = key_df["variable"].map(variable_labels)
key_df["OR"] = key_df["odds_ratio"].round(2)
key_df["95% CI"] = (
    key_df["ci_lower"].round(2).astype(str)
    + "-"
    + key_df["ci_upper"].round(2).astype(str)
)
key_df["p-value"] = key_df["p_value"].apply(format_p_value)
key_df["N"] = key_df["n"].apply(format_n)

summary_table = key_df[[
    "Sample",
    "Term",
    "OR",
    "95% CI",
    "p-value",
    "N"
]]

summary_table.to_csv(
    RESULTS_DIR / "06c_rq3_holdout_summary_table.csv",
    index=False
)

print("\n===== Clean RQ3 summary table =====")
print(summary_table)


# -----------------------------
# 2. Save interaction-only table
# -----------------------------

interaction_df = key_df[
    key_df["variable"] == "mandate_x_US"
].copy()

interaction_table = interaction_df[[
    "Sample",
    "OR",
    "95% CI",
    "p-value",
    "N"
]]

interaction_table.to_csv(
    RESULTS_DIR / "06c_rq3_holdout_interaction_table.csv",
    index=False
)

# -----------------------------
# 3. Forest plot for interaction OR
# -----------------------------

plot_df = interaction_df.copy()
plot_df = plot_df.iloc[::-1].reset_index(drop=True)

y_pos = range(len(plot_df))

fig, ax = plt.subplots(figsize=(8, 4.6))

ax.errorbar(
    plot_df["odds_ratio"],
    y_pos,
    xerr=[
        plot_df["odds_ratio"] - plot_df["ci_lower"],
        plot_df["ci_upper"] - plot_df["odds_ratio"]
    ],
    fmt="o",
    color="#1F4E79",
    ecolor="#1F4E79",
    elinewidth=2,
    capsize=5,
    markersize=7
)

ax.axvline(
    1,
    color="gray",
    linestyle="--",
    linewidth=1
)

ax.set_yticks(y_pos)
ax.set_yticklabels(plot_df["Sample"])
ax.set_xlabel("Odds ratio for mandate period x US interaction")
ax.set_title(
    "RQ3 hold-out robustness check",
    fontsize=13,
    weight="bold"
)

for i, row in plot_df.iterrows():
    label = (
        f"OR {row['odds_ratio']:.2f} "
        f"({row['ci_lower']:.2f}-{row['ci_upper']:.2f}), "
        f"N={int(row['n']):,}"
    )
    ax.text(
        row["ci_upper"] + 0.05,
        i,
        label,
        va="center",
        fontsize=9
    )

x_min = min(0.8, plot_df["ci_lower"].min() - 0.15)
x_max = plot_df["ci_upper"].max() + 0.65

ax.set_xlim(x_min, x_max)
ax.grid(axis="x", linestyle=":", alpha=0.5)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "06c_rq3_holdout_interaction_forest_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -----------------------------
# 4. Performance table and plot
# -----------------------------

perf_df = pd.read_csv(
    RESULTS_DIR / "06b_rq3_holdout_test_performance.csv"
)

perf_clean = perf_df[
    perf_df["metric"].isin(["test_auc", "test_accuracy"])
].copy()

metric_labels = {
    "test_auc": "Test AUC",
    "test_accuracy": "Test accuracy"
}

perf_clean["Metric"] = perf_clean["metric"].map(metric_labels)
perf_clean["Value"] = perf_clean["value"].round(2)

performance_table = perf_clean[["Metric", "Value"]]

performance_table.to_csv(
    RESULTS_DIR / "06c_rq3_holdout_performance_summary.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(5.5, 3.5))

bars = ax.bar(
    performance_table["Metric"],
    performance_table["Value"],
    color=["#2C7FB8", "#7FCDBB"]
)

ax.set_ylim(0, 1)
ax.set_ylabel("Value")
ax.set_title(
    "RQ3 held-out testing performance",
    fontsize=13,
    weight="bold"
)

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.03,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=11
    )

ax.grid(axis="y", linestyle=":", alpha=0.5)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "06c_rq3_holdout_performance_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved clean RQ3 outputs:")
print(RESULTS_DIR / "06c_rq3_holdout_summary_table.csv")
print(RESULTS_DIR / "06c_rq3_holdout_interaction_table.csv")
print(RESULTS_DIR / "06c_rq3_holdout_interaction_forest_plot.png")
print(RESULTS_DIR / "06c_rq3_holdout_performance_summary.csv")
print(RESULTS_DIR / "06c_rq3_holdout_performance_plot.png")