# 4c_rq2_make_holdout_tables_figures.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

# -----------------------------
# Helper formatting functions
# -----------------------------

def format_p_value(p):
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def format_n(n):
    return f"{int(n):,}"


# -----------------------------
# 1. Read RQ2 hold-out OR results
# -----------------------------

or_df = pd.read_csv(
    RESULTS_DIR / "04b_rq2_holdout_within_mandate_results.csv"
)

sample_labels = {
    "full_sample": "Full sample",
    "training_sample": "Training sample",
    "testing_sample": "Testing sample"
}

or_df["Sample"] = or_df["sample"].map(sample_labels)
or_df["OR"] = or_df["odds_ratio"].round(2)
or_df["95% CI"] = (
    or_df["ci_lower"].round(2).astype(str)
    + "-"
    + or_df["ci_upper"].round(2).astype(str)
)
or_df["p-value"] = or_df["p_value"].apply(format_p_value)
or_df["N"] = or_df["n"].apply(format_n)

summary_table = or_df[[
    "Sample",
    "OR",
    "95% CI",
    "p-value",
    "N"
]]

summary_table.to_csv(
    RESULTS_DIR / "04c_rq2_holdout_summary_table.csv",
    index=False
)

print("\n===== Clean summary table =====")
print(summary_table)


# -----------------------------
# 2. Save clean summary table as PNG
# -----------------------------

fig, ax = plt.subplots(figsize=(9, 2.4))
ax.axis("off")

table = ax.table(
    cellText=summary_table.values,
    colLabels=summary_table.columns,
    cellLoc="center",
    colLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.6)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#D9D9D9")
    else:
        cell.set_facecolor("#FFFFFF")
    cell.set_edgecolor("#B0B0B0")

plt.title(
    "RQ2 hold-out robustness check: mandate-period odds ratios",
    fontsize=13,
    weight="bold",
    pad=12
)

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "04c_rq2_holdout_summary_table.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# -----------------------------
# 3. Forest plot for OR and 95% CI
# -----------------------------

plot_df = or_df.copy()
plot_df = plot_df.iloc[::-1].reset_index(drop=True)

y_pos = range(len(plot_df))

fig, ax = plt.subplots(figsize=(8, 4.2))

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
ax.set_xlabel("Odds ratio for sustained mandate period")
ax.set_title(
    "RQ2 hold-out robustness check",
    fontsize=13,
    weight="bold"
)

for i, row in plot_df.iterrows():
    label = f"OR {row['odds_ratio']:.2f} ({row['ci_lower']:.2f}-{row['ci_upper']:.2f}), N={int(row['n']):,}"
    ax.text(
        row["ci_upper"] + 0.05,
        i,
        label,
        va="center",
        fontsize=9
    )

ax.set_xlim(0.8, max(plot_df["ci_upper"]) + 0.65)
ax.grid(axis="x", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "04c_rq2_holdout_or_forest_plot.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# -----------------------------
# 4. Read and clean performance results
# -----------------------------

perf_df = pd.read_csv(
    RESULTS_DIR / "04b_rq2_holdout_test_performance.csv"
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
    RESULTS_DIR / "04c_rq2_holdout_performance_summary.csv",
    index=False
)

print("\n===== Clean performance table =====")
print(performance_table)


# -----------------------------
# 5. Performance bar plot
# -----------------------------

fig, ax = plt.subplots(figsize=(5.5, 3.5))

bars = ax.bar(
    performance_table["Metric"],
    performance_table["Value"],
    color=["#2C7FB8", "#7FCDBB"]
)

ax.set_ylim(0, 1)
ax.set_ylabel("Value")
ax.set_title(
    "Held-out testing performance",
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
    RESULTS_DIR / "04c_rq2_holdout_performance_plot.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


print("\nSaved clean outputs:")
print(RESULTS_DIR / "04c_rq2_holdout_summary_table.csv")
print(RESULTS_DIR / "04c_rq2_holdout_summary_table.png")
print(RESULTS_DIR / "04c_rq2_holdout_or_forest_plot.png")
print(RESULTS_DIR / "04c_rq2_holdout_performance_summary.csv")
print(RESULTS_DIR / "04c_rq2_holdout_performance_plot.png")