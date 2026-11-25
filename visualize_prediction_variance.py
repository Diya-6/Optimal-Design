
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from src.model import model_matrix, information_matrix


# ----------------------------------
# 1. Load optimized designs
# ----------------------------------

RESULTS_DIR = Path("results")

criteria = ["D", "A", "E", "I"]

designs = {}

for criterion in criteria:

    designs[criterion] = pd.read_csv(
        RESULTS_DIR / f"{criterion}_optimal_design.csv"
    ).to_numpy()


# ----------------------------------
# 2. Create prediction grid
# ----------------------------------

levels = np.linspace(-1, 1, 101)

X1, X2 = np.meshgrid(levels, levels)

prediction_grid = np.column_stack([
    X1.ravel(),
    X2.ravel()
])


# ----------------------------------
# 3. Calculate prediction variance
# ----------------------------------

variance_maps = {}

for criterion, design in designs.items():

    M = information_matrix(design)

    # Construct prediction model matrix
    F = model_matrix(prediction_grid)

    # Solve MZ = F.T
    Z = np.linalg.solve(M, F.T)

    # Calculate prediction variance factors
    variance = np.sum(F * Z.T, axis=1)

    # Reshape into a 101 x 101 grid
    variance_maps[criterion] = variance.reshape(
        X1.shape
    )


# ----------------------------------
# 4. Use common contour levels
# ----------------------------------

all_values = np.concatenate([
    values.ravel()
    for values in variance_maps.values()
])

contour_levels = np.linspace(
    all_values.min(),
    all_values.max(),
    20
)


# ----------------------------------
# 5. Plot four contour maps
# ----------------------------------

fig, axes = plt.subplots(
    2, 2,
    figsize=(13, 10)
)

axes = axes.flatten()

for ax, criterion in zip(axes, criteria):

    contour = ax.contourf(
        X1,
        X2,
        variance_maps[criterion],
        levels=contour_levels
    )

    ax.set_title(
        f"{criterion}-Optimal Design"
    )

    ax.set_xlabel("Temperature (coded)")
    ax.set_ylabel("Concentration (coded)")

    ax.set_aspect("equal")

    fig.colorbar(
        contour,
        ax=ax,
        label="Prediction variance factor"
    )


plt.suptitle(
    "Prediction Variance Across Experimental Region",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "prediction_variance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
