
import numpy as np

from src.model import information_matrix
from src.criteria import (
    check_information_matrix,
    d_optimality,
    a_optimality,
    e_optimality,
    i_optimality
)


def objective_score(design, criterion, prediction_grid):
    """
    Calculate only the selected optimality criterion.

    All criteria are converted into maximization problems.
    """

    try:

        # Construct information matrix
        M = information_matrix(design)

        # Check whether the design is valid
        check_information_matrix(M)

        # Calculate only the required objective
        if criterion == "D":
            return d_optimality(M)

        elif criterion == "A":
            return -a_optimality(M)

        elif criterion == "E":
            return e_optimality(M)

        elif criterion == "I":
            return -i_optimality(M, prediction_grid)

        else:
            raise ValueError("Unknown criterion.")

    except (ValueError, np.linalg.LinAlgError):

        # Invalid candidate design
        return -np.inf



def coordinate_exchange(
    criterion,
    prediction_grid,
    n_runs=12,
    candidate_levels=None,
    max_iterations=100,
    seed=42,
    tolerance=1e-8,
    verbose=False
):
    """
    Grid-based coordinate-exchange algorithm.

    Parameters
    ----------
    criterion : str
        D, A, E or I

    prediction_grid : ndarray
        Points used for I-optimality evaluation.

    n_runs : int
        Number of experimental runs.

    candidate_levels : ndarray
        Allowed values for each factor.

    max_iterations : int
        Maximum number of complete coordinate sweeps.

    seed : int
        Random seed for reproducibility.

    tolerance : float
        Minimum objective improvement for accepting a change.

    Returns
    -------
    design : ndarray
        Optimized experimental design.

    history : list
        Objective score after each iteration.
    """

    criterion = criterion.upper()

    if criterion not in ["D", "A", "E", "I"]:
        raise ValueError("Criterion must be D, A, E or I.")

    if candidate_levels is None:
        candidate_levels = np.linspace(-1, 1, 21)

    rng = np.random.default_rng(seed)

    # ----------------------------------
    # 1. Generate a valid initial design
    # ----------------------------------

    while True:

        design = rng.uniform(
            -1,
            1,
            size=(n_runs, 2)
        )

        initial_score = objective_score(
            design,
            criterion,
            prediction_grid
        )

        if np.isfinite(initial_score):
            break

    current_score = initial_score

    history = [current_score]

    # ----------------------------------
    # 2. Coordinate-exchange iterations
    # ----------------------------------

    for iteration in range(max_iterations):

        previous_score = current_score

        # Visit each experimental run
        for i in range(n_runs):

            # Visit each factor
            for j in range(2):

                best_value = design[i, j]
                best_score = current_score

                # Try all candidate levels
                for value in candidate_levels:

                    trial_design = design.copy()

                    trial_design[i, j] = value

                    score = objective_score(
                        trial_design,
                        criterion,
                        prediction_grid
                    )

                    # Accept an improving coordinate
                    if score > best_score + tolerance:

                        best_score = score
                        best_value = value

                # Update the actual design
                design[i, j] = best_value

                current_score = best_score

        # Save objective after one complete sweep
        history.append(current_score)

        improvement = current_score - previous_score

        if verbose:
            print(
                f"Iteration {iteration + 1}: "
                f"Score = {current_score:.8f}"
            )

        # ----------------------------------
        # 3. Check convergence
        # ----------------------------------

        if improvement <= tolerance:
            if verbose:
                print("Convergence achieved.")
            break

    return design, history
