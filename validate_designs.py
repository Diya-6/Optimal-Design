
import numpy as np
import pandas as pd

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

levels = np.linspace(-1, 1, 21)

X1, X2 = np.meshgrid(levels, levels)

prediction_grid = np.column_stack([
    X1.ravel(),
    X2.ravel()
])

F_prediction = model_matrix(prediction_grid)


# ----------------------------------
# 3. Independently validate each design
# ----------------------------------

validation_results = []

for criterion, design in designs.items():

    # Construct model matrix
    X = model_matrix(design)

    # Construct information matrix
    M = information_matrix(design)

    # Matrix rank
    rank = np.linalg.matrix_rank(X)

    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(M)

    # D-optimality
    sign, logdet = np.linalg.slogdet(M)

    D = logdet if sign > 0 else np.nan

    # A-optimality
    A = np.trace(np.linalg.solve(
        M, np.eye(M.shape[0])
    ))

    # E-optimality
    E = eigenvalues.min()

    # I-optimality
    Z = np.linalg.solve(M, F_prediction.T)

    prediction_variance = np.sum(
        F_prediction * Z.T,
        axis=1
    )

    I = np.mean(prediction_variance)

    validation_results.append({

        "Design": criterion + "-optimal",

        "Runs": len(design),

        "Parameters": X.shape[1],

        "Rank": rank,

        "Minimum Eigenvalue": E,

        "Log Determinant": D,

        "A Criterion": A,

        "I Criterion": I,

        "Valid": rank == X.shape[1]
    })


# ----------------------------------
# 4. Display results
# ----------------------------------

validation_df = pd.DataFrame(
    validation_results
)

print("\nSTATISTICAL VALIDATION")
print(validation_df.round(6).to_string(index=False))


# ----------------------------------
# 5. Save results
# ----------------------------------

validation_df.to_csv(
    RESULTS_DIR / "validation_results.csv",
    index=False
)

print("\nValidation completed!")
