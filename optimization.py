
import numpy as np
import pandas as pd

from src.coordinate_exchange import coordinate_exchange
from src.criteria import calculate_all_criteria


def run_multiple_starts(
    criterion,
    prediction_grid,
    n_starts=10,
    n_runs=12,
    candidate_levels=None,
    max_iterations=100,
    base_seed=42,
    verbose=False
):
    """
    Run coordinate-exchange from multiple
    random initial designs.

    Returns
    -------
    best_result : dict
        Best design, its history and criterion values.

    run_results : list
        Results from every initialization.
    """

    criterion = criterion.upper()

    if criterion not in ["D", "A", "E", "I"]:
        raise ValueError("Invalid criterion.")

    run_results = []

    best_result = None
    best_score = -np.inf

    # ----------------------------------
    # Run multiple initializations
    # ----------------------------------

    for run in range(n_starts):

        seed = base_seed + run

        print(
            f"\n{criterion}-optimality | "
            f"Starting run {run + 1}/{n_starts}"
        )

        # Run coordinate-exchange
        design, history = coordinate_exchange(
            criterion=criterion,
            prediction_grid=prediction_grid,
            n_runs=n_runs,
            candidate_levels=candidate_levels,
            max_iterations=max_iterations,
            seed=seed,
            verbose=verbose
        )

        # Final transformed objective score
        final_score = history[-1]

        # Store this run
        run_results.append({
            "run": run + 1,
            "seed": seed,
            "initial_score": history[0],
            "final_score": final_score,
            "iterations": len(history) - 1,
            "design": design.copy(),
            "history": history.copy()
        })

        print(
            f"Final objective score: {final_score:.8f}"
        )

        # ----------------------------------
        # Select the best result
        # ----------------------------------

        if final_score > best_score:

            best_score = final_score

            best_result = {
                "criterion": criterion,
                "design": design.copy(),
                "score": final_score,
                "history": history.copy(),
                "seed": seed
            }

    # ----------------------------------
    # Calculate all criteria for best design
    # ----------------------------------

    best_result["criteria"] = calculate_all_criteria(
        best_result["design"],
        prediction_grid
    )

    return best_result, run_results


def summarize_runs(run_results):
    """
    Convert multiple-start results into
    a Pandas DataFrame.
    """

    rows = []

    for result in run_results:

        rows.append({
            "Run": result["run"],
            "Seed": result["seed"],
            "Initial Score": result["initial_score"],
            "Final Score": result["final_score"],
            "Iterations": result["iterations"]
        })

    return pd.DataFrame(rows)
