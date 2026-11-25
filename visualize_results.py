
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ----------------------------------
# 1. Define project paths
# ----------------------------------

RESULTS_DIR = Path("results")

criteria = ["D", "A", "E", "I"]

designs = {}


# ----------------------------------
# 2. Load optimized designs
# ----------------------------------

for criterion in criteria:

    file_path = RESULTS_DIR / f"{criterion}_optimal_design.csv"

    designs[criterion] = pd.read_csv(file_path).to_numpy()


# ----------------------------------
# 3. Plot experimental designs
# ----------------------------------

fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 10)
)

axes = axes.flatten()

for ax, criterion in zip(axes, criteria):

    design = designs[criterion]

    x1 = design[:, 0]
    x2 = design[:, 1]

    # Plot experimental points
    ax.scatter(
        x1,
        x2,
        s=100,
        marker="o",
        edgecolors="black",
        alpha=0.8
    )

    # Label experimental runs
    for i, (x, y) in enumerate(zip(x1, x2)):

        ax.annotate(
            str(i + 1),
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8
        )

    # Experimental region
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    ax.set_xticks(np.linspace(-1, 1, 5))
    ax.set_yticks(np.linspace(-1, 1, 5))

    ax.set_xlabel("Temperature (coded)")
    ax.set_ylabel("Concentration (coded)")

    ax.set_title(f"{criterion}-Optimal Design")

    ax.grid(True, alpha=0.3)

    ax.set_aspect("equal")

plt.suptitle(
    "Comparison of D-, A-, E- and I-Optimal Designs",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "experimental_designs.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
