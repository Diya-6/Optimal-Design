
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


RESULTS_DIR = Path("results")


# ----------------------------------
# 1. Load comparison table
# ----------------------------------

df = pd.read_csv(
    RESULTS_DIR / "criteria_comparison.csv",
    index_col=0
)

criteria = ["D", "A", "E", "I"]

df = df[criteria]


# ----------------------------------
# 2. Normalize criterion values
# ----------------------------------

normalized = pd.DataFrame(
    index=df.index,
    columns=df.columns,
    dtype=float
)

for criterion in criteria:

    values = df[criterion]

    minimum = values.min()
    maximum = values.max()

    if np.isclose(maximum, minimum):
        normalized[criterion] = 1.0

    elif criterion in ["D", "E"]:

        # Higher is better
        normalized[criterion] = (
            values - minimum
        ) / (maximum - minimum)

    else:

        # Lower is better
        normalized[criterion] = (
            maximum - values
        ) / (maximum - minimum)


# ----------------------------------
# 3. Plot heatmap
# ----------------------------------

fig, ax = plt.subplots(
    figsize=(9, 6)
)

image = ax.imshow(
    normalized.to_numpy(dtype=float),
    aspect="auto",
    vmin=0,
    vmax=1
)

ax.set_xticks(range(len(criteria)))
ax.set_xticklabels(criteria)

ax.set_yticks(range(len(df.index)))
ax.set_yticklabels(df.index)

ax.set_xlabel("Optimality criterion")
ax.set_ylabel("Optimized design")

ax.set_title(
    "Comparative Performance of Optimal Designs"
)

# Display normalized values inside cells
for i in range(len(df.index)):

    for j in range(len(criteria)):

        value = normalized.iloc[i, j]

        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center"
        )


fig.colorbar(
    image,
    ax=ax,
    label="Normalized performance (higher is better)"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "comparative_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Save normalized table
normalized.to_csv(
    RESULTS_DIR / "normalized_comparison.csv"
)

print("\nNormalized comparison:")
print(normalized.round(3))
