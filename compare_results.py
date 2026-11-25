
import pandas as pd

from src.optimization import run_multiple_starts

import numpy as np


# Create prediction grid
levels = np.linspace(-1, 1, 21)

X1, X2 = np.meshgrid(levels, levels)

prediction_grid = np.column_stack([
    X1.ravel(),
    X2.ravel()
])


criteria = ["D", "A", "E", "I"]

comparison = {}

for criterion in criteria:

    best_result, _ = run_multiple_starts(
        criterion=criterion,
        prediction_grid=prediction_grid,
        n_starts=10,
        n_runs=12,
        max_iterations=100,
        base_seed=42
    )

    comparison[criterion] = best_result["criteria"]


# Convert results into a DataFrame
comparison_df = pd.DataFrame(
    comparison
).T

comparison_df.index.name = "Design"

print("\nCOMPARISON OF FOUR OPTIMAL DESIGNS")
print(comparison_df.round(6))


# Save comparison
comparison_df.to_csv(
    "results/criteria_comparison.csv"
)
