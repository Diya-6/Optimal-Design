import numpy as np
import matplotlib.pyplot as plt

from src.coordinate_exchange import coordinate_exchange
from src.criteria import calculate_all_criteria


# ----------------------------------
# 1. Create prediction grid
# ----------------------------------

levels = np.linspace(-1, 1, 21)

X1, X2 = np.meshgrid(levels, levels)

prediction_grid = np.column_stack([
    X1.ravel(),
    X2.ravel()
])


# ----------------------------------
# 2. Run D-optimal coordinate exchange
# ----------------------------------

design, history = coordinate_exchange(
    criterion="D",
    prediction_grid=prediction_grid,
    n_runs=12,
    max_iterations=100,
    seed=42
)


# ----------------------------------
# 3. Display optimized design
# ----------------------------------

print("\nOptimized D-optimal design:")
print(design)


# ----------------------------------
# 4. Calculate final criteria
# ----------------------------------

results = calculate_all_criteria(
    design,
    prediction_grid
)

print("\nFinal optimality criteria:")

for name, value in results.items():
    print(f"{name}: {value:.8f}")


# ----------------------------------
# 5. Plot convergence
# ----------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    range(len(history)),
    history,
    marker="o"
)

plt.xlabel("Iteration")
plt.ylabel("D-optimal objective (log determinant)")

plt.title("Coordinate-Exchange Convergence")

plt.grid(True)

plt.tight_layout()

plt.show()
