
import numpy as np
import pandas as pd

from pathlib import Path


RESULTS_DIR = Path("results")

df = pd.read_csv(
    RESULTS_DIR / "benchmark_comparison.csv",
    index_col=0
)


# ----------------------------------
# 1. Define comparisons
# ----------------------------------

comparisons = {

    "D": ("D-optimal", "Face-centered CCD"),

    "A": ("A-optimal", "Face-centered CCD"),

    "E": ("E-optimal", "Face-centered CCD"),

    "I": ("I-optimal", "Face-centered CCD")
}


# ----------------------------------
# 2. Calculate improvements
# ----------------------------------

results = []

for criterion, (optimized, benchmark) in comparisons.items():

    opt_value = df.loc[optimized, criterion]

    base_value = df.loc[benchmark, criterion]

    if criterion == "D":

        improvement = np.exp(
            opt_value - base_value
        )

        description = "Determinant ratio"

    elif criterion == "E":

        improvement = (
            (opt_value - base_value) / base_value
        ) * 100

        description = "Percentage increase"

    else:

        improvement = (
            (base_value - opt_value) / base_value
        ) * 100

        description = "Percentage reduction"

    results.append({

        "Criterion": criterion,

        "Optimized value": opt_value,

        "CCD value": base_value,

        "Improvement": improvement,

        "Measure": description
    })


improvement_df = pd.DataFrame(results)

print("\nIMPROVEMENT OVER CCD")

print(improvement_df.round(4).to_string(index=False))

improvement_df.to_csv(
    RESULTS_DIR / "benchmark_improvements.csv",
    index=False
)
