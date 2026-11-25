import numpy as np

from src.model import model_matrix
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

print("Number of prediction points:", len(prediction_grid))

# ----------------------------------
# 2. Generate a valid initial design
# ----------------------------------

rng = np.random.default_rng(42)

while True:

    design = rng.uniform(
        low=-1,
        high=1,
        size=(12, 2)
    )

    X = model_matrix(design)

    if np.linalg.matrix_rank(X) == 6:
        break


print("\nExperimental Design:")
print(design)


# ----------------------------------
# 3. Calculate all four criteria
# ----------------------------------

results = calculate_all_criteria(
    design,
    prediction_grid
)


# ----------------------------------
# 4. Display results
# ----------------------------------

print("\nOPTIMALITY CRITERIA")
print("---------------------------")

for criterion, value in results.items():

    print(f"{criterion}-optimality: {value:.6f}")