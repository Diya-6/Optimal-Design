
import numpy as np
import pandas as pd

from src.optimization import run_multiple_starts, summarize_runs


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
# 2. Optimization settings
# ----------------------------------

CRITERIA = ["D", "A", "E", "I"]

N_STARTS = 10
N_RUNS = 12

CANDIDATE_LEVELS = np.linspace(-1, 1, 21)

MAX_ITERATIONS = 100


# ----------------------------------
# 3. Store all optimized designs
# ----------------------------------

best_designs = {}

all_run_results = {}


# ----------------------------------
# 4. Optimize each criterion
# ----------------------------------

for criterion in CRITERIA:

    print("\n" + "=" * 50)
    print(f"OPTIMIZING {criterion}-OPTIMAL DESIGN")
    print("=" * 50)

    best_result, run_results = run_multiple_starts(
        criterion=criterion,
        prediction_grid=prediction_grid,
        n_starts=N_STARTS,
        n_runs=N_RUNS,
        candidate_levels=CANDIDATE_LEVELS,
        max_iterations=MAX_ITERATIONS,
        base_seed=42
    )

    best_designs[criterion] = best_result
    # Save convergence history of the best run

    history_df = pd.DataFrame({
        "Iteration": range(len(best_result["history"])),
        "Objective Score": best_result["history"]
    })

    history_df.to_csv(
        f"results/{criterion}_convergence.csv",
        index=False
    )

    all_run_results[criterion] = run_results

    print("\nBEST DESIGN FOUND:")
    print(best_result["design"])

    print("\nALL FOUR CRITERIA:")
    print(best_result["criteria"])


# ----------------------------------
# 5. Display multi-start summary
# ----------------------------------

for criterion in CRITERIA:

    print("\n", criterion, "- OPTIMIZATION SUMMARY")

    summary = summarize_runs(
        all_run_results[criterion]
    )

    print(summary.to_string(index=False))


# ----------------------------------
# 6. Save optimized designs
# ----------------------------------

for criterion, result in best_designs.items():

    np.savetxt(
        f"results/{criterion}_optimal_design.csv",
        result["design"],
        delimiter=",",
        header="Temperature,Concentration",
        comments=""
    )


print("\nAll four optimal designs saved successfully!")

import pandas as pd

# ----------------------------------
# Compare all four best designs
# ----------------------------------

comparison_rows = []

for criterion, result in best_designs.items():

    row = {
        "Design": criterion + "-optimal"
    }

    row.update(result["criteria"])

    comparison_rows.append(row)


comparison_df = pd.DataFrame(comparison_rows)

comparison_df = comparison_df.set_index("Design")

print("\nFINAL COMPARISON")
print(comparison_df.round(6))


# Save comparison table
comparison_df.to_csv(
    "results/criteria_comparison.csv"
)


# ----------------------------------
# Save multiple-start summaries
# ----------------------------------

for criterion, runs in all_run_results.items():

    summary = summarize_runs(runs)

    summary.to_csv(
        f"results/{criterion}_multiple_starts.csv",
        index=False
    )

print("\nAll results and summaries saved!")