# 7_make_combined_holdout_performance_table.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

# -----------------------------
# Read RQ2 performance
# -----------------------------

rq2_perf = pd.read_csv(
    RESULTS_DIR / "04b_rq2_holdout_test_performance.csv"
)

rq2_auc = rq2_perf.loc[
    rq2_perf["metric"] == "test_auc",
    "value"
].iloc[0]

rq2_accuracy = rq2_perf.loc[
    rq2_perf["metric"] == "test_accuracy",
    "value"
].iloc[0]

# -----------------------------
# Read RQ3 performance
# -----------------------------

rq3_perf = pd.read_csv(
    RESULTS_DIR / "06b_rq3_holdout_test_performance.csv"
)

rq3_auc = rq3_perf.loc[
    rq3_perf["metric"] == "test_auc",
    "value"
].iloc[0]

rq3_accuracy = rq3_perf.loc[
    rq3_perf["metric"] == "test_accuracy",
    "value"
].iloc[0]

# -----------------------------
# Build clean table
# -----------------------------

combined_table = pd.DataFrame({
    "Model": [
        "RQ2: US model",
        "RQ3: AUS-US model"
    ],
    "Test AUC": [
        round(rq2_auc, 2),
        round(rq3_auc, 2)
    ],
    "Test accuracy": [
        round(rq2_accuracy, 2),
        round(rq3_accuracy, 2)
    ]
})

combined_table.to_csv(
    RESULTS_DIR / "07_combined_holdout_performance_table.csv",
    index=False
)

print("\n===== Combined hold-out performance table =====")
print(combined_table)

# -----------------------------
# Save table as PNG
# -----------------------------

fig, ax = plt.subplots(figsize=(8, 2.2))
ax.axis("off")

table = ax.table(
    cellText=combined_table.values,
    colLabels=combined_table.columns,
    cellLoc="center",
    colLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.7)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#D9D9D9")
    else:
        cell.set_facecolor("#FFFFFF")
    cell.set_edgecolor("#B0B0B0")

plt.title(
    "Held-out testing performance",
    fontsize=14,
    weight="bold",
    pad=12
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "07_combined_holdout_performance_table.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved to:")
print(RESULTS_DIR / "07_combined_holdout_performance_table.csv")
print(RESULTS_DIR / "07_combined_holdout_performance_table.png")