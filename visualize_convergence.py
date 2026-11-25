
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


RESULTS_DIR = Path("results")

criteria = ["D", "A", "E", "I"]


# ----------------------------------
# Plot convergence curves
# ----------------------------------

fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 9)
)

axes = axes.flatten()

for ax, criterion in zip(axes, criteria):

    history = pd.read_csv(
        RESULTS_DIR / f"{criterion}_convergence.csv"
    )

    ax.plot(
        history["Iteration"],
        history["Objective Score"],
        marker="o"
    )

    ax.set_title(
        f"{criterion}-Optimal Convergence"
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective score")

    ax.grid(True, alpha=0.3)


plt.suptitle(
    "Coordinate-Exchange Convergence",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "convergence_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
