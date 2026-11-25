
import numpy as np
import pandas as pd

from pathlib import Path

from src.model import model_matrix, information_matrix


RESULTS_DIR = Path("results")

N_RUNS = 12
N_RANDOM = 100

# ----------------------------------
# 1. Prediction grid
# ----------------------------------

levels = np.linspace(-1, 1, 21)

X1, X2 = np.meshgrid(levels, levels)

prediction_grid = np.column_stack([
    X1.ravel(),
    X2.ravel()
])

F_prediction = model_matrix(prediction_grid)


# ----------------------------------
# 2. Define evaluation function
# ----------------------------------

def evaluate_design(design):

    X = model_matrix(design)

    M = information_matrix(design)

    rank = np.linalg.matrix_rank(X)

    if rank < X.shape[1]:
        raise ValueError("Design is rank deficient.")

    eigenvalues = np.linalg.eigvalsh(M)

    sign, logdet = np.linalg.slogdet(M)

    if sign <= 0:
        raise ValueError("Invalid information matrix.")

    A = np.trace(
        np.linalg.solve(M, np.eye(M.shape[0]))
    )

    E = eigenvalues.min()

    Z = np.linalg.solve(M, F_prediction.T)

    prediction_variance = np.sum(
        F_prediction * Z.T,
        axis=1
    )

    I = prediction_variance.mean()

    return {
        "D": logdet,
        "A": A,
        "E": E,
        "I": I
    }


# ----------------------------------
# 3. Three-level factorial design
# ----------------------------------

factor_levels = [-1, 0, 1]

factorial = np.array([
    [x1, x2]
    for x1 in factor_levels
    for x2 in factor_levels
])

center = np.array([
    [0, 0],
    [0, 0],
    [0, 0]
])

three_level_design = np.vstack([
    factorial,
    center
])


# ----------------------------------
# 4. Face-centered CCD
# ----------------------------------

factorial_points = np.array([
    [-1, -1],
    [-1, 1],
    [1, -1],
    [1, 1]
])

axial_points = np.array([
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1]
])

center_points = np.array([
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0]
])

ccd_design = np.vstack([
    factorial_points,
    axial_points,
    center_points
])


# ----------------------------------
# 5. Load optimized designs
# ----------------------------------

designs = {

    "D-optimal": pd.read_csv(
        RESULTS_DIR / "D_optimal_design.csv"
    ).to_numpy(),

    "A-optimal": pd.read_csv(
        RESULTS_DIR / "A_optimal_design.csv"
    ).to_numpy(),

    "E-optimal": pd.read_csv(
        RESULTS_DIR / "E_optimal_design.csv"
    ).to_numpy(),

    "I-optimal": pd.read_csv(
        RESULTS_DIR / "I_optimal_design.csv"
    ).to_numpy(),

    "Three-level factorial": three_level_design,

    "Face-centered CCD": ccd_design
}


# ----------------------------------
# 6. Generate random designs
# ----------------------------------

rng = np.random.default_rng(123)

candidate_levels = np.linspace(-1, 1, 21)

random_results = []

while len(random_results) < N_RANDOM:

    random_design = rng.choice(
        candidate_levels,
        size=(N_RUNS, 2)
    )

    try:

        metrics = evaluate_design(random_design)

        random_results.append({
            "design": random_design,
            "metrics": metrics
        })

    except (ValueError, np.linalg.LinAlgError):

        continue


# ----------------------------------
# 7. Select best random design
#    for each criterion
# ----------------------------------

for criterion in ["D", "A", "E", "I"]:

    if criterion in ["D", "E"]:
        best_index = np.argmax([
            result["metrics"][criterion]
            for result in random_results
        ])

    else:
        best_index = np.argmin([
            result["metrics"][criterion]
            for result in random_results
        ])

    best_random = random_results[best_index]

    designs[
        f"Best random ({criterion})"
    ] = best_random["design"]


# ----------------------------------
# 8. Evaluate all designs
# ----------------------------------

rows = []

for name, design in designs.items():

    metrics = evaluate_design(design)

    row = {
        "Design": name,
        "Runs": len(design)
    }

    row.update(metrics)

    rows.append(row)


comparison = pd.DataFrame(rows)

comparison = comparison.set_index("Design")

print("\nBENCHMARK COMPARISON")

print(comparison.round(6))


# ----------------------------------
# 9. Save results
# ----------------------------------

comparison.to_csv(
    RESULTS_DIR / "benchmark_comparison.csv"
)

print("\nBenchmarking completed!")
